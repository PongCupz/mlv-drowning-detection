from utility.restnest50 import RestNet50

rest_net = RestNet50.RestNet50()
rest_net.test('data/datasets/mlv/validation/drowning/*',"weights-drowning2")