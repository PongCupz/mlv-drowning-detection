from utility.restnest50 import RestNet50

rest_net = RestNet50.RestNet50("data/datasets/mlv/")
rest_net.test('data/datasets/mlv/validation/drowning/*')