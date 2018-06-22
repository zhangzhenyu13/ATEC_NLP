def split(inputfile,ratio):
    lines=[]
    with open(inputfile,"r") as f:
        for line in f:
            lines.append(line)

    import random
    n_count=len(lines)
    print("shuffle %d records"%n_count)
    random.shuffle(lines)
    train=lines[:int(ratio*n_count)]
    test=lines[int(ratio*n_count):]
    with open("./data/train.csv","w") as f:
        for line in train:
            f.write(line)
    with open("./data/test.csv","w") as f:
        for line in test:
            f.write(line)


if __name__ == '__main__':
    split("./data/train_nlp_data.csv",0.8)