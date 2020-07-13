%============================================
% 单通道图像呈现彩图效果
% 其他通道用全黑图表示
%===============================================
clc
clearvars
close all

tic

[FileName, FilePath]=uigetfile('*.jpg;*.png;*.tif;*.img;*.bmp;','请选择一幅参考图片');
image_name = [FilePath FileName];

Iorg = imread(image_name);
[rI, cI] = size(Iorg);

Ir = zeros(rI, cI, 3);

%-- 输入对话框，选择呈现颜色　--%
% 1-红色; 2-绿色; 3-蓝色
prompt = {'请输入呈现颜色（1红色; 2绿色; 3蓝色）'};
title = '单通道图像呈现彩图效果';
lines = [1]';
def = {'1'};
Channel_input = inputdlg(prompt, title, lines, def);
Nc = str2double(Channel_input);
%-----------------------------%

Ir(:,:,Nc) = Iorg;

%-- 保存图像　--%
rf = [FilePath, 'RGB-', FileName];
imwrite(uint8(Ir), rf);
%--------------%












toc
