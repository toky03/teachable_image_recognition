import cv2
import os

base_folder = '.'


def create_folder_if_not_exists(path_name):
    if not os.path.exists(path_name):
        os.mkdir(path_name)


def prepare_folders():
    create_folder_if_not_exists(base_folder+'/image_set')
    create_folder_if_not_exists(base_folder+'/image_set/label_a')
    create_folder_if_not_exists(base_folder+'/image_set/label_b')
    create_folder_if_not_exists(base_folder+'/image_set/label_c')
    create_folder_if_not_exists(base_folder+'/image_set/label_l')


def collect_pictures(video):
    counter = 0
    while True:
        ret, frame = video.read()
        frame = cv2.rotate(frame,cv2.ROTATE_180)
        frame = cv2.resize(frame, (640, 640))
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            cv2.imwrite(base_folder+'/image_set/label_a/{}.png'.format(str(counter)), frame)
            print('image a saved')
            counter += 1
        elif key == ord('s'):
            cv2.imwrite(base_folder+'/image_set/label_b/{}.png'.format(str(counter)), frame)
            print('image b saved')
            counter += 1
        elif key == ord('d'):
            cv2.imwrite(base_folder+'/image_set/label_c/{}.png'.format(str(counter)), frame)
            print('image c saved')
            counter += 1
        elif key == ord('l'):
            cv2.imwrite(base_folder+'/image_set/label_l/{}.png'.format(str(counter)), frame)
            print('image l saved')
            counter += 1


if __name__ == '__main__':
    prepare_folders()
    vid = cv2.VideoCapture(0)
    print('Collect Train dataset')
    collect_pictures(vid)
    vid.release()
    cv2.destroyAllWindows()
