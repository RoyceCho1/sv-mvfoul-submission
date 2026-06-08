from getpass import getpass

from SoccerNet.Downloader import SoccerNetDownloader as SNdl


def main():
    password = getpass("SoccerNet password: ")

    downloader = SNdl(LocalDirectory="data/SoccerNet")
    downloader.downloadDataTask(
        task="mvfouls",
        split=["train", "valid", "test", "challenge"],
        password=password,
        version="720p",
    )


if __name__ == "__main__":
    main()
