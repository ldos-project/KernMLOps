import random
import string

TIME_SLOT_TO_SEC = 100

def generate_random_string(length=10):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(length))

def generate_sleep_time_args(min_sleep_slots: int, max_sleep_slots: int) -> str:
    # sample sleep time in range [0, max_sleep_slots]
    sleep_time_slots = random.randint(min_sleep_slots, max_sleep_slots)
    return [""] * sleep_time_slots

def generate_stress_ng_args(remaining_time_slots: int) -> tuple[int, float]:
    # sample ops in range [min_ops, max_ops]
    num_stressors = random.randint(1, 32)
    # ops = random.randint(1 * num_stressors, 10000)
    timeout = random.randint(1, remaining_time_slots) * 0.01

    return num_stressors, timeout

def stressor_to_cmd_args(stressor_name: str, num_stressors: int) -> str:
    return f"--{stressor_name} {num_stressors}"

def stress_ng_args_to_cmd(timeout: float, stressor_config: dict[str, int]) -> str:
    cmd_args = [f"timeout {timeout} stress-ng"]
    for stressor_name, num_stressors in stressor_config.items():
        cmd_args.append(stressor_to_cmd_args(stressor_name, num_stressors))
    return " ".join(cmd_args)

def write_random_strings_to_file(filename, arg_list: list[str]):
    with open(filename, 'w') as file:
        for args in arg_list:
            file.write(args + '\n')

if __name__ == "__main__":
    for i in range(10000):
        filename = f'stress-ng-args/{i+1}.txt'
        num_time_slots = 200  # Change this to the desired number of lines

        arg_list = []
        sleep_min = 0
        while len(arg_list) < num_time_slots:
            sleep_max = sleep_min + random.randint(10, 90)
            arg_list.extend(generate_sleep_time_args(sleep_min, sleep_max))
            if len(arg_list) >= num_time_slots:
                break
            num_stressors, timeout = generate_stress_ng_args(num_time_slots - len(arg_list))
            arg_list.append(stress_ng_args_to_cmd('iomix', num_stressors, timeout))
            sleep_min = int(timeout * 100)
            # arg_list.append(generate_stress_ng_args('iomix', num_time_slots - len(arg_list)))

        arg_list = arg_list[:num_time_slots]

        write_random_strings_to_file(filename, arg_list)
