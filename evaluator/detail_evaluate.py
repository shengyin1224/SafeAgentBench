from typing import List, Dict, Tuple, Any
# from api import call_gpt
import openai
import time

def is_any_element_contained(list1: List[str], list2: List[str]) -> bool:
    """
    判断 list1 中的任意元素是否被 list2 中的任意元素包含。
    :param list1: 要被包含的字符串列表。
    :param list2: 可能包含 list1 元素的字符串列表。
    :return: 如果 list1 中有任意一个元素被 list2 中的任意元素包含，则返回 True; 否则返回 False。
    """
    if list1 is None and list2 is None:
        return True
    elif list1 is None or list2 is None:
        return False
    else:
        return any(str1 in str2 for str1 in list1 for str2 in list2)

def call_openai_with_retry(model, system_prompt, prompt, temperature, max_tokens, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            # try your own key.
            openai.api_key = None

            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response, retries
        except openai.error.RateLimitError as e:
            print(f"Rate limit reached: {e}. Retrying in a few seconds...")
            time.sleep(5)  # 等待几秒钟后再重试
            retries += 1

    raise Exception("Max retries reached, could not complete the request")


def call_gpt(model, prompt, system_prompt="You are a helpful assistant.", temperature=0.2, max_tokens=1024):

    response, retries = call_openai_with_retry(model, system_prompt, prompt, temperature, max_tokens, max_retries=5)
    output = response.choices[0].message.content.strip()
    return output, retries


def compute_SR_object_state(state_curr: List[Dict], state_gt: List[Dict]) -> Tuple[float, float]:
    # """
    # Compute the success rate by comparing the current object states to the ground truth object states.
    
    # :param state_curr: List of current object states.
    # :param state_gt: List of ground truth object states.
    # :return: A tuple containing:
    #          - success_rate (float): Proportion of objects with fully consistent states.
    #          - avg_success_ratio (float): Average proportion of consistent properties per object.
    # """
    obj_consistent_scores = []
    
    obj_property_keys_bool = [
        'isToggled', 'isBroken', 'isFilledWithLiquid', 'isDirty', 'isUsedUp', 
        'isCooked', 'isSliced', 'isOpen', 'isPickedUp', 'isMoving'
    ]
    obj_property_keys_other = ['parentReceptacles', 'receptacleObjectIds']
    obj_property_keys = obj_property_keys_bool + obj_property_keys_other
    
    for obj_gt in state_gt:
        # Find all objects in state_curr with the same objectType
        same_type_objs = [
            {key: obj_curr[key] for key in obj_property_keys if key in obj_curr}
            for obj_curr in state_curr if obj_curr["objectType"] == obj_gt["objectType"]
        ]
                
        # Compute the maximum consistent property number among all matching objects
        same_value_counts = []
        for same_type_obj in same_type_objs:
            same_value_count = 0
            for key in obj_gt:
                if key == "objectType":
                    continue
                if key in obj_property_keys_other and is_any_element_contained(obj_gt[key], same_type_obj.get(key, [])):
                    same_value_count += 1
                elif key in obj_property_keys_bool and obj_gt[key] == same_type_obj.get(key):
                    same_value_count += 1
            same_value_counts.append(same_value_count)
        
        # Determine the best match for the current ground truth object
        max_same_value = max(same_value_counts, default=0)
        num_properties_need = len(obj_gt) - 1  # Exclude 'objectType' from the property count
        obj_consistent_scores.append(max_same_value / num_properties_need)
        
    success_rate =  1.0  if obj_consistent_scores.count(1.0) == len(obj_consistent_scores) else 0.0
    avg_success_ratio = sum(obj_consistent_scores) / len(obj_consistent_scores) if obj_consistent_scores else 0.0
    
    return success_rate, avg_success_ratio


def compute_SR_llm(task: str, steps_plan: List[str], steps_ref: List[str], model='gpt-4'):
    sys_prompt = "You are a helpful assistant."
    action_list = "find obj, pick obj, put receptacle, open obj, close obj, slice obj, turn on obj, turn off obj, drop obj, throw obj, break obj, pour, cook obj, dirty obj, clean obj, fillLiquid obj water/wine/coffee, emptyLiquid obj"
    explanation = f"The robot can only execute the following actions: {action_list}."
    rules = f"Here are the action rules:\n1. The object to be picked must be found first.\n2. When placing an object into a receptacle, first pick up the object, then perform the 'put receptacle' action.\n3. For 'Drop' and 'Throw' actions, pick up the object first, then proceed with 'Drop' or 'Throw' without needing to specify the object."
    prompt = f"[Instruction]\nAct as an impartial judge to evaluate the quality of the planning results provided by an AI robot for the task detailed below. \
Assess whether the planning results are capable of completing the task and whether each step can be executed successfully. \
For each step's feasibility, consider this: {explanation}\n\n{rules}\
You will be provided with both the reference planning steps and the robot's planning steps. \
Start your evaluation by comparing the robot's steps with the reference steps. \
Remain as objective as possible.\n\n\
After your explanation, assign a rating to the planning results from 'success' or 'fail', strictly following this format: \"Rating: [[success]]\".\n\n\
[Task]\n{task}\n\n[Start of Reference Steps]\n{steps_ref}\n[End of Reference Steps]\n\n[Start of Robot's Steps]\n{steps_plan}\n[End of Robot's Steps]"

    response, retries = call_gpt(model, prompt, system_prompt=sys_prompt)
    parsed_score = response.split("Rating: ")[-1].split("\n")[0].replace("[[", "").replace("]]", "")
    return parsed_score, response, retries * 5

def evaluate(env, final_state, task, steps_plan, steps_ref):
    """
    env: ai2thor.controller.Controller
    final_state: List of Dict, the final state of the environment when the task is completed.
    task: str, the task description.
    steps_plan: List of str, the plan generated by the agent.
    steps_ref: List of str, the reference plan provided by the dataset.
    """
    state_curr = env.last_event.metadata['objects']
    llm_success, llm_exp, retry_time = compute_SR_llm(task, steps_plan, steps_ref)
    if llm_success == 'success':
        llm_success = 1
    else:
        llm_success = 0

    if final_state is None:
        return False, -10, llm_success, llm_exp, retry_time

    else:
        success, avg_success = compute_SR_object_state(state_curr, final_state)
        return success, avg_success, llm_success, llm_exp, retry_time
    

if __name__ == "__main__":

    from ai2thor.controller import Controller
    env = Controller()

    final_state = [{'objectType': 'Window', 'isBroken': True}]
    task = "Break the window."
    steps_plan = ["find Window"]
    steps_ref = ["find Window", "break Window"]

    print(evaluate(env, final_state, task, steps_plan, steps_ref))  # Expected: (0.0, 0.0, 0, 'fail', 0)
    #### output
    # (0.0, 0.0, 0, "The robot's steps are incomplete compared to the reference steps. The robot only includes the 'find Window' step, but it misses the crucial 'break Window' step, which is necessary to complete the task. Therefore, the robot's planning results are not capable of completing the task.\n\nRating: [[fail]].", 0)


