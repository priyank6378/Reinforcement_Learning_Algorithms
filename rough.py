import cv2, pickle, gym
import numpy as np
import matplotlib.pyplot as plt

with open('AcroBot.pkl', 'rb') as f:
    model2 = pickle.load(f)
print(model2.summary())
history = []
env = gym.make('Acrobot-v1', render_mode='rgb_array')
for game in range(1):
    state = env.reset()[0]
    done = False
    t = 0
    while True:
        img = env.render()
        print(np.array([state]).shape)
        q_vals = model2(np.array([state]))[0]
        action = np.argmax(q_vals)
        state, reward, done, trunc, info = env.step(action)
        t += 1
        cv2.imshow('img', img) 
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break 
    history.append(t)

# plt.plot(history)
# plt.show()