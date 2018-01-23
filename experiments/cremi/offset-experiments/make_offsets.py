import json


#
# just an example for making the default offsets
#
def get_default_offsets():
    return [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
            [-2, 0, 0], [0, -3, 0], [0, 0, -3],
            [-3, 0, 0], [0, -9, 0], [0, 0, -9],
            [-4, 0, 0], [0, -27, 0], [0, 0, -27]]


if __name__ == '__main__':
    default_offsets = get_default_offsets()
    with open('./default_offsets.json', 'w') as f:
        json.dump(default_offsets, f)
