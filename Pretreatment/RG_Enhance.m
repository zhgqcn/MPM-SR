file_path_r =  'D:/ALL_DataSet/R_G_Partition/R_Part/train_target/';% 图像文件夹路径
file_path_g =  'D:/ALL_DataSet/R_G_Partition/G_Part/train_target_1024_128/';% 图像文件夹路径
img_path_list_r = dir(strcat(file_path_r,'*.tif'));%获取该文件夹中所有tif格式的图像
img_path_list_g = dir(strcat(file_path_g,'*.tif'));%获取该文件夹中所有tif格式的图像
img_num = length(img_path_list_r);%获取图像总数量
if img_num > 0 %有满足条件的图像
        for k = 1:img_num %逐一读取图像
            image_name_r = img_path_list_r(k).name;% 
            image_name_g = img_path_list_g(k).name;% 图像名
            
            imgr  =  imread(strcat(file_path_r,image_name_r));
            imgg  =  imread(strcat(file_path_g,image_name_g));
            black =  imread('D:/PycharmDOC/test_photo/all_black.tif');
            
            x = cat(3, imgr, imgg, imgg);

            Img_R_path = strcat('D:/ALL_DataSet/RGGE/train/RGGE_' ,image_name_r);
            imwrite(x ,Img_R_path);
        end
end  