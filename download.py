import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory="./videos/soccernet/")

mySoccerNetDownloader.downloadDataTask(task="reid", split=["train", "valid", "test", "challenge"])

# mySoccerNetDownloader.downloadDataTask(task="reid-2023", split=["train", "valid", "test", "challenge"])

# mySoccerNetDownloader.downloadDataTask(task="tracking", split=["train", "test", "challenge"])

# mySoccerNetDownloader.downloadDataTask(task="tracking-2023", split=["train", "test", "challenge"])