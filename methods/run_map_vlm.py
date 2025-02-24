import sys
sys.path.append('/home/ubuntu/xhpang/ai2thor')
from map_vlm import Agents
from ai2thor.controller import Controller
from low_level_planner import LowLevelPlanner
from utils import *
from our_evaluate import compute_SR_object_state
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import jsonlines
import os
import time
import threading

lock = threading.Lock()



def ndarray_to_base64(img_array: np.ndarray, image_format: str = "PNG") -> str:
    """
    将 numpy.ndarray 格式的图像转换为 Base64 编码格式。

    参数:
    img_array (np.ndarray): 输入的图像数组。
    image_format (str): 图像格式（默认为 "PNG"）。

    返回:
    str: 纯 Base64 编码的图像字符串，不包含前缀。
    """
    # 确保图像数组是 uint8 类型
    img_array = img_array.astype(np.uint8)

    # 将 numpy 数组转换为 PIL 图像
    image = Image.fromarray(img_array)

    # 将 PIL 图像保存到字节流中
    buffered = io.BytesIO()
    image.save(buffered, format=image_format)

    # 将字节流转换为 Base64 编码
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # 返回纯 Base64 编码图像字符串
    return img_str


def run_map(task, scene='FloorPlan1', low_level_plan=None):
    controller = Controller(scene=scene)
    objs_all = all_objs(controller)
    # print('All objects:', objs_all)
    st_time = time.time()
    
    if low_level_plan is None:
    
        # event = controller.step(
        #     action="AddThirdPartyCamera",
        #     position=dict(x=-1.25, y=1, z=-1),
        #     rotation=dict(x=0, y=0, z=0),
        #     fieldOfView=120
        # )
        # img = event.third_party_camera_frames[0]
        
        img = controller.last_event.frame
        # save_ndarray_as_image(img, 'test.png')
        
        img = ndarray_to_base64(img)
        agent = Agents(img, task)
        env_info, plan = agent.multi_agent_vision_planning(objs_all)
        if "sorry" in plan:
            return None, None, None, None
        print('Plan:', plan)
        low_level_plan = gen_low_level_plan(plan)
        
    # print(low_level_plan)
    planner = LowLevelPlanner(controller)
    metadata_curr, sr_step = execute_low_level_plan(low_level_plan, planner)
    objs_curr = metadata_curr['objects']
    
    ed_time = time.time()
    time_task = ed_time - st_time
    
    controller.stop()
    del controller
    
    return objs_curr, low_level_plan, time_task, sr_step


def process_data(d, save_path, model="gpt-4o-mini"):
    task = d['instruction']
    scene = d['scene_name']
    obj_gt = d.get('final_state', None)
    
    low_level_plan = d.get('low_level_plan', None)

    objs_curr, low_level_plan, time_task, sr_step = run_map(task, scene=scene, low_level_plan=low_level_plan)

    refusal = False
    if objs_curr is None and low_level_plan is None and time_task is None and sr_step is None:
        refusal = True
        success_rate = 0
        avg_success_ratio = 0
    elif obj_gt is None:
        success_rate = None
        avg_success_ratio = None
    else:
        success_rate, avg_success_ratio = compute_SR_object_state(objs_curr, obj_gt)
        
    # print('Success rate:', success_rate)
    # print('Average success ratio:', avg_success_ratio)
    
    # save low level plan
    with lock:
        with open(save_path, 'a') as f:
            d['low_level_plan'] = low_level_plan
            d['objects_current'] = objs_curr
            d['success_rate'] = success_rate
            d['avg_success_ratio'] = avg_success_ratio
            d['time_task'] = time_task
            d['sr_step'] = sr_step
            d["refusal"] = refusal
            print("Writing to file")
            f.write(json.dumps(d) + '\n')
    
    return success_rate



if __name__ == '__main__':
    
    data_dict = {
        "unsafe": "/home/ubuntu/xhpang/ai2thor/dataset_final/instructions_300_final_version1-115checked.jsonl",
        "safe": "/home/ubuntu/xhpang/ai2thor/dataset_final/safe_instructions_step.jsonl",
        "abstract": "/home/ubuntu/xhpang/ai2thor/dataset_final/instructions_abstract100.jsonl",
        "horizon": "/home/ubuntu/xhpang/ai2thor/dataset_final/instructions_long_horizon.jsonl"
    }
    task = "safe"
    data = load_dataset(data_dict, name=task)[:50]    
    model = "gpt-4"
    save_path = f'/home/ubuntu/xhpang/ai2thor/0913/mat_{model}_{task}.jsonl'
    
    # load processed data
    if os.path.exists(save_path):
        with jsonlines.open(save_path) as reader:
            processed_data = list(reader)
        processed_data = [d['instruction'] for d in processed_data]
        data = [d for d in data if d['instruction'] not in processed_data]
    print("Remaining tasks:", len(data))
    
    success_rates = []
    max_threads = 40 # Adjust the number of threads as needed

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        future_to_data = {executor.submit(process_data, d, save_path): d for d in data}
        
        for future in as_completed(future_to_data):

            try:
                result = future.result()
                if result is not None:
                    success_rates.append(result)
            except Exception as exc:
                # raise exc
                print(f'Generated an exception: {exc}')
                traceback.print_exc()  # 打印完整的异常堆栈信息
    
    print('Success rates:', success_rates)
