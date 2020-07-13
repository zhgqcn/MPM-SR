%======================================================
% MPM灰度均衡
% f1=imadjust（f，[low_in  high_in],[low_out  high_out],gamma）
%=======================================================

clc
close all
clearvars
tic

[FileName, FilePath]=uigetfile('*.jpg;*.png;*.tif;*.img;*.bmp;','请选择一幅参考图片');
image_name = [FilePath FileName];

Iorg = imread(image_name);
[rI, cI] = size(Iorg(:,:,1));

%-- 判断图像信息所在的通道 --%
ifocus = zeros(1,3);      % 需要处理的分量，初始值为0，对应分量有值时，元素变为1  [0,0,0]
for i=1:3   % i=1红色 ; i=2绿色; i=3蓝色 
   Inidraw = Iorg(:,:,i);    %通道读取
   Meani = mean(Inidraw(:)); %先把矩阵转化为向量，然后计算均值
   if(Meani>1)  % 该分量有信息，理论上应该大于0就说明有信息，但是为了去噪，设成略大的值
       ifocus(i) = 1;
   end
end
%---------------------------%

[r,c,v] = find(ifocus~=0);

%%
% 只处理有信息的分量
Ir = zeros(rI, cI, 3); 
for i=2
    Ipi = Iorg(:,:,c(i));
    figure;
    subplot(2,2,1);
    imshow(uint8(Ipi));title('1024绿色单通道');
    
    original512 = imread('D:\MATLAB\bin\IMAGES\R_G_SingleChannels\G_channel\original_512\gc_512_1-512pix-speed7-ave1.tif')
    subplot(2,2,2)
    imshow(original512);title('512绿色单通道')
    
    %-- ！！！！整体加亮！！！！ --%
    CL = 54;   
    Ipi_CL = Ipi + CL;
    subplot(2,2,3);
    imshow(uint8(Ipi_CL));title('整体加亮效果');
    %-----------------------------%
    
    %-- ！！！！非线性灰度拉伸！！！！--%
    % gamma < 1 低灰度区强拉伸，高灰度区弱拉伸甚至压缩
    Ipi_histequ = imadjust(Ipi_CL, [57/255, 150/255], [40/255, 255/255], 0.85);
    subplot(2,2,4)
    imshow(uint8(Ipi_histequ));title('非线性灰度拉伸');
    %---------------------------------%
    
    Ir(:,:,i) = Ipi_histequ;
   
    %-- 灰度直方图(原图) --%
    %[count_pi, x_pi] = imhist(Ipi);   % xn是灰度值，count_n是出现的次数
    %[count_ri, x_ri] = imhist(Ir(:,:,i));
    %f_pi = count_pi/(rI*cI);          % 出现次数归一化
    %f_ri = count_ri/(rI*cI);

    figure
    subplot(2,2,1);
%     stem(x_pi, f_pi, 'b.');
    imhist(Ipi);
    title('灰度直方图-原图');
    
    subplot(2,2,2);
    imhist(Ipi_CL);title('灰度直方图-整体加亮');
    
    subplot(2,2,3);
%     stem(x_ri, f_ri, 'r.');
    imhist(uint8(Ir(:,:,i)));
    title('灰度直方图-均衡后');
    %-----------------------------%
   
    %-----------------------------%
    figure
    IpiShow = zeros(rI,cI,3);
    IpiShow(:,:,i) = Ipi;
    imshow(uint8(IpiShow));title('原始RGB图');   

    figure
    IriShow = zeros(rI,cI,3);
    IriShow(:,:,i) = Ipi_histequ;
    imgs = uint8(IriShow);
    imshow(uint8(IriShow));title('增亮＋非线性拉伸RGB图');
    %-----------------------------%
    
    %-- 保存图像　--%
    switch i
        case 1
            ChannelName = 'R-';
        case 2
            ChannelName = 'G-';
        case 3
            ChannelName = 'B-';
    end
    
    rfpi = [FilePath, 'org-', ChannelName, FileName];
    imwrite(uint8(IpiShow), rfpi);
    
    rfri = [FilePath, 'Heq-', ChannelName, FileName];
    I1 = imgs(:,:,1);
    imwrite(I1, rfri);
    %--------------%
end








toc