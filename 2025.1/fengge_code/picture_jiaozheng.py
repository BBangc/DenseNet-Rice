import cv2
import numpy as np


def jiaozheng(img):

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 设置红色阈值
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)

    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)

    # 合并所有红色区域
    mask = mask1 + mask2

    # 寻找红色标记的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centers = []

    # 假设最大的四个区域是红色标记点
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:4]:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append([cx, cy])

    # 确保找到了四个标记点
    if len(centers) != 4:
        raise ValueError("未能找到四个红色标记点")

    # 转换点可能需要排序以匹配目标图像的角
    centers = np.array(centers, dtype="float32")

    # 对找到的四个点进行排序
    # 按y坐标排序
    centers = centers[np.argsort(centers[:, 1])]

    # 选择上面的两个点，并按x坐标排序
    top_two = centers[:2][np.argsort(centers[:2][:, 0])]
    # 选择下面的两个点，并按x坐标排序
    bottom_two = centers[2:][np.argsort(centers[2:][:, 0])]

    # 合并排序后的四个点
    sorted_centers = np.array([top_two[0], top_two[1], bottom_two[0], bottom_two[1]], dtype="float32")

    # 目标点（图像的四个角）
    H_rows, W_cols = img.shape[:2]
    pts2 = np.float32([[0, 0], [W_cols, 0], [0, H_rows], [W_cols, H_rows]])

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(sorted_centers, pts2)
    dst = cv2.warpPerspective(img, M, (W_cols, H_rows))

    return dst

    # # 显示结果
    # cv2.imshow("result", dst)
    # # 保存图像
    # cv2.imwrite('D:\\Processed Grain\\data_set\\OriginData\\transformed_image3.png', dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



