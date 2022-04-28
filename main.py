import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os


def process_single_image(img_path):
    if os.path.isfile(f):
        img = cv.imread(f)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        rows = img_gray.shape[0]
        img_gray = cv.medianBlur(img_gray, 5)
        circles = cv.HoughCircles(img_gray, cv.HOUGH_GRADIENT, 1, rows / 4,
                                  param1=25, param2=15, minRadius=900, maxRadius=1600)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            center = (circles[0][0][0], circles[0][0][1])
            radius = circles[0][0][2]
            print("Found circles: ", circles)

            # Create masks for slices in four different directions
            mask1 = np.zeros(img_gray.shape, np.uint8)
            cv.rectangle(mask1, center, (center[0] + radius, center[1] + 1), 255, -1)
            mask2 = np.zeros(img_gray.shape, np.uint8)
            cv.rectangle(mask2, center, (center[0] + 1, center[1] + radius), 255, -1)
            mask3 = np.zeros(img_gray.shape, np.uint8)
            cv.rectangle(mask3, center, (center[0] - radius, center[1] + 1), 255, -1)
            mask4 = np.zeros(img_gray.shape, np.uint8)
            cv.rectangle(mask4, center, (center[0] + 1, center[1] - radius), 255, -1)
            masked1 = cv.cvtColor(cv.bitwise_and(img, img, mask=mask1), cv.COLOR_BGR2GRAY)
            masked2 = cv.cvtColor(cv.bitwise_and(img, img, mask=mask2), cv.COLOR_BGR2GRAY)
            masked3 = cv.cvtColor(cv.bitwise_and(img, img, mask=mask3), cv.COLOR_BGR2GRAY)
            masked4 = cv.cvtColor(cv.bitwise_and(img, img, mask=mask4), cv.COLOR_BGR2GRAY)

            # Crop out the four slices
            mx, my = masked2.shape
            print(mx, my)
            cropped2 = np.zeros((1, my))
            cropped4 = np.zeros((1, my))
            cropped1 = masked1[center[1]:center[1] + 1, center[0]:center[0] + radius]
            cropped2 = masked2[center[1]:center[1] + radius, center[0]:center[0] + 1].transpose()
            cropped3 = np.flip(masked3[center[1]:center[1] + 1, center[0] - radius:center[0]])
            cropped4 = masked4[center[1]:center[1] - radius, center[0]:center[0] + 1].transpose()

            # Calculate the average of the slices
            nx, ny = cropped2.shape
            print(nx, ny)
            cropped1 = cropped1[:, 0:ny]
            cropped3 = cropped3[:, 0:ny]
            cropped_full = np.array([cropped1, cropped2, cropped3, cropped4])
            cropped_full_average = np.average(cropped_full, axis=0)

            # Construct X-axis and fit the average slice to a third degree polynomial
            xaxis = np.arange(0, cropped_full_average.shape[1])
            xaxis = xaxis.reshape(1, ny)
            full_dataset = np.vstack((xaxis, cropped_full_average))
            full_dataset = full_dataset.transpose()
            DF = pd.DataFrame(full_dataset)
            x1 = np.array(DF[0])
            y1 = np.array(DF[1])
            z = np.polyfit(x1, y1, 3)

            # Dirty trick to get rid of the third dimension
            masked3 = cv.resize(masked3, (0, 0), fx=1, fy=1)

            # Construct a foreground based on the average slice
            foreground = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
            for j in range(0, radius):
                cv.circle(foreground, center, j, int(z[0] * j ** 3 + z[1] * j ** 2 + z[2] * j + z[3] - subval), 2)

            # Dirty trick to get rid of the third dimension
            img_gray = cv.resize(img_gray, (0, 0), fx=1, fy=1)
            foreground = cv.resize(foreground, (0, 0), fx=1, fy=1)

            subtraction = foreground - img_gray

            # TODO: add the scale modifier to separate zones instead of the hardcoded 0.8
            cv.circle(subtraction, (int(center[0]), int(center[1])), int(radius * 0.8), 0, 2)

            mask_sub = np.zeros(img_gray.shape, np.uint8)
            cv.circle(mask_sub, (int(center[0]), int(center[1])), int(radius * 0.8), 255, -1)

            mask_sub_s = np.zeros(img_gray.shape, np.uint8)
            cv.circle(mask_sub_s, (int(center[0]), int(center[1])), int(radius * 0.8) - 10, 255, -1)

            masked_sub = cv.bitwise_and(subtraction, subtraction, mask=mask_sub)
            masked_white = cv.bitwise_not(subtraction, mask=mask_sub_s)

            canny = cv.Canny(masked_white, 100, 200)
            masked_canny = cv.bitwise_and(canny, canny, mask=mask_sub)

            contours, hierarchy = cv.findContours(masked_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            masked_sub_color = cv.cvtColor(masked_canny, cv.COLOR_GRAY2BGR)

            # draw the two different zones
            cv.circle(masked_sub_color, (int(center[0]), int(center[1])), int(12 * cc), (255, 0, 0), 2)
            cv.circle(masked_sub_color, (int(center[0]), int(center[1])), int(19.2 * cc), (255, 0, 0), 2)

            first_zone_defect_counter = 0
            second_zone_defect_counter = 0

            for cnt in contours:
                M = cv.moments(cnt)
                if M['m00'] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    area = cv.contourArea(cnt)
                    cv.drawContours(masked_sub_color, cnt, 0, (0, 0, 255), 1)
                    xx, yy, ww, hh = cv.boundingRect(cnt)
                    defect_size = int(math.sqrt(ww ** 2 + hh * 2))

                    cv.rectangle(masked_sub_color, (xx, yy), (xx + ww, yy + hh), (255, 0, 0), 1)

                    if math.sqrt((cX - center[0]) ** 2 + (cY - center[1]) ** 2) < 12 * cc:
                        if defect_size > cc * 0.294:
                            cv.putText(masked_sub_color, str(defect_size), (xx, yy - 30), cv.FONT_HERSHEY_SIMPLEX, 2,
                                       invalid, 1, cv.LINE_AA)
                        else:
                            if defect_size > cc * 0.0735:
                                if first_zone_defect_counter < 2:
                                    cv.putText(masked_sub_color, str(defect_size), (xx, yy - 30),
                                               cv.FONT_HERSHEY_SIMPLEX, 2, valid, 1, cv.LINE_AA)
                                    first_zone_defect_counter += 1
                                else:
                                    cv.putText(masked_sub_color, str(defect_size), (xx, yy - 30),
                                               cv.FONT_HERSHEY_SIMPLEX, 2, invalid, 1, cv.LINE_AA)
                            else:
                                cv.putText(masked_sub_color, str(defect_size), (xx, yy - 30),
                                           cv.FONT_HERSHEY_SIMPLEX, 2, valid, 1, cv.LINE_AA)

                    elif math.sqrt((cX - center[0]) ** 2 + (cY - center[1]) ** 2) < 19.2 * cc:
                        if defect_size > cc * 0.588:
                            cv.putText(masked_sub_color, str(defect_size), (xx, yy - 30), cv.FONT_HERSHEY_SIMPLEX, 2,
                                       invalid, 1, cv.LINE_AA)
                        else:
                            if defect_size > cc * 0.1:
                                if second_zone_defect_counter < 2:
                                    cv.putText(masked_sub_color, str(defect_size), (xx, yy - 30),
                                               cv.FONT_HERSHEY_SIMPLEX, 2, valid, 1, cv.LINE_AA)
                                    second_zone_defect_counter += 1
                                else:
                                    cv.putText(masked_sub_color, str(defect_size), (xx, yy - 30),
                                               cv.FONT_HERSHEY_SIMPLEX, 2, invalid, 1, cv.LINE_AA)
                            else:
                                cv.putText(masked_sub_color, str(defect_size), (xx, yy - 30),
                                           cv.FONT_HERSHEY_SIMPLEX, 2, valid, 1, cv.LINE_AA)
                    print(area)
            return masked_sub_color
        else:
            return None


if __name__ == '__main__':
    font = cv.FONT_HERSHEY_DUPLEX

    # conversion coefficient, px/mm
    cc = 50.6
    # subtraction value
    subval = 15
    # if we should go through all files in the input directory
    process_everything = True
    # settings for the tag colors
    valid = (0, 255, 0)
    invalid = (0, 0, 255)
    # input and output directories
    input_directory = "./data/images/"
    output_directory = "./data/output/"
    single_image = "00224.jpg"

    if process_everything:
        for filename in os.listdir(input_directory):
            f = os.path.join(input_directory, filename)
            print(f)
            processed_image = process_single_image(f)
            fout = os.path.join(output_directory, os.path.basename(filename))
            if processed_image is not None:
                cv.imwrite(fout, processed_image)
            else:
                print("No circles found, output file not created")
    else:
        f = os.path.join(input_directory, single_image)
        print(f)
        processed_image = process_single_image(f)
        if processed_image is not None:
            cv.namedWindow("subtraction", cv.WINDOW_NORMAL)
            cv.resizeWindow("subtraction", 800, 800)
            cv.imshow("subtraction", processed_image)
            cv.imwrite(os.path.join(output_directory, single_image), processed_image)
        else:
            print("No circles found, output file not created")

    cv.waitKey(0)
