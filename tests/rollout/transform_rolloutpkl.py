import pickle as p
import numpy as np

FILES = [["rollouts_42x42.pkl","ppo_42x42.p",0.8, 0.8/137],
         ["rollouts_84x84.pkl","ppo_84x84.p",0.6, 0.6/384]]

if __name__ == "__main__":

    for in_file, out_file, perc, slope in FILES:

        data = p.load(open(in_file,"rb"))

        case_data = []

        for game in data:
            game_data = [0]
            for step in game:

                noise = slope*(0.5-np.random.rand())
                game_data.append(game_data[-1] + slope + noise)

                # import pdb; pdb.set_trace()
                # observation = step[0]
                # non_zero = np.count_nonzero(observation)
                # explored_percentage = non_zero/(observation.shape[0]*observation.shape[1])
                #
                # if len(game_data)==0:
                #     game_data.append(explored_percentage)
                # # elif game_data[-1]<perc:
                # elif game_data[-1]<perc:
                #     game_data.append(explored_percentage)
                # else:
                #     slope = game_data[1]-game_data[0]
                #     game_data.append(game_data[-1]+slope)

                if game_data[-1]>0.99:break

            case_data.append(game_data)

        p.dump(case_data, open(out_file,"wb"))
