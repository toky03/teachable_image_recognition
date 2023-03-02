import cv2
import os


def create_folder_if_not_exists(path_name):
    if not os.path.exists(path_name):
        os.mkdir(path_name)


def prepare_folders():
    create_folder_if_not_exists('image_set')
    create_folder_if_not_exists('image_set/label_a')
    create_folder_if_not_exists('image_set/label_b')


def collect_pictures(video):
    counter = 0
    while True:
        ret, frame = video.read()
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            cv2.imwrite('image_set/label_a/{}.png'.format(str(counter)), frame)
            counter += 1
        elif key == ord('b'):
            cv2.imwrite('image_set/label_b/{}.png'.format(str(counter)), frame)
            counter += 1


if __name__ == '__main__':
    prepare_folders()
    vid = cv2.VideoCapture(0)
    print('Collect Train dataset')
    collect_pictures(vid)
    vid.release()
    cv2.destroyAllWindows()
