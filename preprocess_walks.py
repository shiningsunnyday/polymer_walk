import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    args = parser.parse_args()
    lines = open(args.data).readlines()
    walks = []
    for line in lines:
        line = line.rstrip('\n')
        if 'X' in line:
            if 'X=' not in line.split()[-1]:
                breakpoint()
            ind = line.find('X=')+2
            temp = line.split()[-2]
            for val in line[ind:].split(','):
                walks.append(temp.replace('X', val))
        else:
            walks.append(line.split(' ')[-1])
    with open(args.data.replace('.txt', '_preprocess.txt'), 'w+') as f:
        for w in walks:
            f.write(f"{w}\n")