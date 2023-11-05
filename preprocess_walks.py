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
            if 'X=' not in line.split()[2]:
                breakpoint()
            ind = line.split()[2].find('X=')+2            
            temp = line.split()[1]
            X_subs = line.split()[2][ind:].split(';')
            if len(X_subs) != len(line.split()[3:]):
                breakpoint()
            else:
                for X_sub, prop_val in zip(X_subs, line.split()[3:]):
                    walks.append(temp.replace('X', X_sub)+' '+prop_val)

        else:
            if len(line.split()[2:]) != 1:
                breakpoint()
            prop_val = line.split()[2:][0]
            walks.append(line.split(' ')[1]+' '+prop_val)
    with open(args.data.replace('.txt', '_preprocess.txt'), 'w+') as f:
        for l, w in zip(lines, walks):
            f.write(f"{l.split()[0]} {w}\n")
            