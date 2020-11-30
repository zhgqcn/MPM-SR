%% crop the im into 256*256

clear;clc;
file_path = 'D:\ALL_DataSet\BSDS\train_HR_4\';              % 设定你存放图片的目录
img_path_list = dir(strcat(file_path, '*.png')); % 选后缀为 .png 的图片
img_num = length(img_path_list); %获得图片数量

for j = 1:img_num 
    image_name = img_path_list(j).name;
    image = imread(strcat(file_path, image_name));
    [m,n]=size(image);
    if m == 481
        crop_image = imcrop(image, [0, 0, 320, 320]); % 使用 imcrop() 函数来裁剪图片，第二个参数的格式为 [XMIN YMIN WIDTH HEIGHT]
    else
        crop_image = imcrop(image, [0, 0, 320, 320]); % 使用 imcrop() 函数来裁剪图片，第二个参数的格式为 [XMIN YMIN WIDTH HEIGHT]
    end
    
    imwrite(crop_image, strcat('D:\ALL_DataSet\BSDS200\train_HR_4\', image_name)); % 保存文件
end
