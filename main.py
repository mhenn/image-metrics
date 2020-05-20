from frames import * 
from metrics import * 




frame = getFrames('videos/my_video-2.mkv', 1)[0]
cmpFrame = getFrames('videos/newest_test.mp4', 1)[0]

m = Metric()

m.run([frame],[cmpFrame])

m.run([frame],[cmpFrame])
m.run([frame],[cmpFrame])


print(m.metrics.keys())

