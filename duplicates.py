def duplicates(path):
    d = duplicate
    for entry in os.scandir(path):
        if entry.is_file():
            file = entry.name
            try:
                os.mkdir(path + "/" + d)
            except FileExistsError:
                if os.path.exists(path + "/" +  d):
                    os.rename(path +  "/" + file, path + "/" + d +  "/" + file)
                    print("sent to " + d)
                    print("similar image exists")
                else:
                    raise
