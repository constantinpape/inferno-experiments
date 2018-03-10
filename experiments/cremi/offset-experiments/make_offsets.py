import json


#
# just an example for making the default offsets
#
def get_default_offsets():
    return [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
            [-2, 0, 0], [0, -3, 0], [0, 0, -3],
            [-3, 0, 0], [0, -9, 0], [0, 0, -9],
            [-4, 0, 0], [0, -27, 0], [0, 0, -27]]


def get_mws_offsets():
    return [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
            [0, -9, 0], [0, 0, -9], [0, -9, -9], [0, 9, -9],
            [0, -9, 4], [0, -4, -9], [0, 4, -9], [0, 9, -4],
            [-2, 0, 0], [-2, 0, -9], [-2, -9, 0], [-2, 9, -9], [-2, -9, -9],
            [-3, 0, 0], [0, -27, 0], [0, 0, -27]]


if __name__ == '__main__':
    mws_offsets = get_mws_offsets()
    with open('./mws_offsets.json', 'w') as f:
        json.dump(mws_offsets, f)
