from utility.restnest50 import RestNet50

rest_net = RestNet50.RestNet50()
rest_net.train("data/datasets/mlv/","weights-drowning2")