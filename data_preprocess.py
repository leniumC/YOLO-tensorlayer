import cv2
import os


def main():
    root_dir = 'stage1_train'
    new_dir = 'new_dataset'

    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)

    count = 0
    total = len(os.listdir(root_dir))

    for id in os.listdir(root_dir):
        count += 1
        print('Completed images: %d/%d' % (count, total))

        img_path = os.path.join(root_dir, id, 'images', id + '.png')
        mask_dir = os.path.join(root_dir, id, 'masks')

        img_orig = cv2.imread(img_path)
        orig_size = str(img_orig.shape[0]) + ',' + str(img_orig.shape[1]) + ',0,0'  # added 0's for easier csv reading
        img_resized = cv2.resize(img_orig, (256, 256))

        cv2.imwrite(os.path.join(new_dir, id + '.png'), img_resized)

        with open(os.path.join(new_dir, id + '.txt'), 'w') as f:
            f.write(orig_size)
            for mask_name in os.listdir(mask_dir):
                mask_path = os.path.join(mask_dir, mask_name)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (256, 256))

                points = cv2.findNonZero(mask)
                rect = cv2.boundingRect(points)

                s = [str(x) for x in rect]
                s_txt = '\n' + s[0] + ',' + s[1] + ',' + s[2] + ',' + s[3]

                f.write(s_txt)

                # cv2.rectangle(img_resized, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), [255, 255, 0], 1)

        # cv2.imshow('bounded', img_resized)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
