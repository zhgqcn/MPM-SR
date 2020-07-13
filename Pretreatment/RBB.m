file_path =  'D:/ALL_DataSet/R_G_Partition/R_Part/train_target/';% 图像文件夹路径
img_path_list = dir(strcat(file_path,'*.tif'));%获取该文件夹中所有jpg格式的图像
img_num = length(img_path_list);%获取图像总数量
if img_num > 0 %有满足条件的图像
        for k = 1:img_num %逐一读取图像
            image_name = img_path_list(k).name;% 图像名
            img  =  imread(strcat(file_path,image_name));

            black = imread('D:/PycharmDOC/test_photo/all_black.tif');
            
            x = cat(3, img, black, black);

            Img_R_path = strcat('D:/ALL_DataSet/RBB/train/RBB_' ,image_name);
            imwrite(x ,Img_R_path);
        end
end