import random
import string
import numpy as np

TIME_SLOT_TO_SEC = 100

def generate_random_string(length=10):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(length))

def generate_sleep_time_args(max_sleep_slots: int) -> str:
    # sample sleep time in range [0, max_sleep_slots]
    sleep_time_slots = random.randint(0, max_sleep_slots)
    return [""] * sleep_time_slots

def generate_stress_ng_args(stressor_name: str, remaining_time_slots: int) -> str:
    # sample ops in range [min_ops, max_ops]
    num_stressors = random.randint(1, 32)
    ops = random.randint(1 * num_stressors, 512 * num_stressors)
    timeout = int(remaining_time_slots / TIME_SLOT_TO_SEC) + 1

    return f"--{stressor_name} {num_stressors} --{stressor_name}-ops {ops} --timeout {timeout}s"

def write_random_strings_to_file(filename, arg_list: list[str]):
    with open(filename, 'w') as file:
        for args in arg_list:
            file.write(args + '\n')

if __name__ == "__main__":
    for i in range(10):
        filename = f'stress-ng-args/{i}.txt'
        num_time_slots = 200  # Change this to the desired number of lines

        arg_list = []
        while len(arg_list) < num_time_slots:
            arg_list.extend(generate_sleep_time_args(200))
            arg_list.append(generate_stress_ng_args('memcpy', num_time_slots - len(arg_list)))

        arg_list = arg_list[:num_time_slots]

        write_random_strings_to_file(filename, arg_list)
