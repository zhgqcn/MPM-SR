file_path =  'D:/MATLAB/bin/IMAGES/R_G_SingleChannels/G_channel/original_1024/';% 图像文件夹路径
img_path_list = dir(strcat(file_path,'*.tif'));%获取该文件夹中所有jpg格式的图像
img_num = length(img_path_list);%获取图像总数量
if img_num > 0 %有满足条件的图像
        for k = 1:img_num %逐一读取图像
            image_name = img_path_list(k).name;% 图像名
            Ipi  =  imread(strcat(file_path,image_name));

%             figure;
%             subplot(2,2,1);
%             imshow(uint8(Ipi));title('1024红色单通道');
% 
%             original512 = imread('D:\MATLAB\bin\IMAGES\R_channel\original512\rc_512_1-512pix-speed7-ave1.tif')
%             subplot(2,2,2)
%             imshow(original512);title('512红色单通道')

            %-- ！！！！整体加亮！！！！ --%
            CL = 54;   
            Ipi_CL = Ipi + CL;
%             subplot(2,2,3);
%             imshow(uint8(Ipi_CL));title('整体加亮效果');
            %-----------------------------%

            %-- ！！！！非线性灰度拉伸！！！！--%
            % gamma < 1 低灰度区强拉伸，高灰度区弱拉伸甚至压缩
            Ipi_histequ = imadjust(Ipi_CL, [57/255, 150/255], [40/255, 255/255], 0.85);
%             subplot(2,2,4)
%             imshow(uint8(Ipi_histequ));title('非线性灰度拉伸');

            Img_R_path = strcat('D:/MATLAB/bin/IMAGES/R_G_Enhanced/G_Channel/G_target_1024_enhanced/' ,image_name);
            imwrite(Ipi_histequ ,Img_R_path);
        end
end