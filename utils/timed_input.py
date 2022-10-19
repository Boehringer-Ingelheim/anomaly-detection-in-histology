import sys
import select
import logging


def limited_time_input(message, delay):

    print(message)

    i, _, _ = select.select( [sys.stdin], [], [], delay)

    if i:
        read = sys.stdin.readline().strip()
    else:
        logging.info("skipping use input")
        #print("skipping use input")
        read = ""

    return read

if __name__ == "__main__":
    choice = limited_time_input("please input your choice", 10)
    if not choice:
        print('nothing was chosen')
    else:
        print('{} was chosen'.format(choice))