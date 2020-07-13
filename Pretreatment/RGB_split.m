file_path =  'D:/MATLAB/bin/IMAGES/Original_512/';% 图像文件夹路径
img_path_list = dir(strcat(file_path,'*.tif'));%获取该文件夹中所有jpg格式的图像
img_num = length(img_path_list);%获取图像总数量
if img_num > 0 %有满足条件的图像
        for j = 1:img_num %逐一读取图像
            image_name = img_path_list(j).name;% 图像名
            Image =  imread(strcat(file_path,image_name));
       
            I1=Image(:,:,2);%G通道

            
            Img_R_path = strcat('D:/MATLAB/bin/IMAGES/G_channel/original_512/' , 'gc_512_', image_name);
       
            %保存结果

            imwrite(I1,Img_R_path);

        end
end
