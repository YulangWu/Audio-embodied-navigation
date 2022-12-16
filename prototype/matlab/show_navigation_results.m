clear all;close all;clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                     The University of Texas at Dallas
%                              Yulang Wu
%                             2022.12.13
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot some figures for illustration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%info from iGibson about Rs_int scene:
%xmin,ymin,xmax,ymax: 
%-4.269999980926514 -4.392959117889404 2.940000057220459 3.847040891647339
% max_length = 5
% floor_map = 2*max_length,2*max_length
% x_new = (x+max_length)*100
% y_new = (y+max_length)*100
% x_new = x_new/
% y_new = y_new/2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%read the source and robot positions for training and test from Python results:
max_length = 5;%model size is 5m by 5m
amplifier = 2; %grid length is 2cm


train_pos_src = dlmread('train_src_pos.txt');
train_pos_src = reshape(train_pos_src,2,length(train_pos_src)/2).';
len_train_src = length(train_pos_src);

test_pos_src = dlmread('test_src_pos.txt');
test_pos_src = reshape(test_pos_src,2,length(test_pos_src)/2).';
len_test_src = length(test_pos_src);
test_pos_src_in_scene = zeros(len_test_src,2);
for i = 1:len_test_src
    test_pos_src_in_scene(i,1) = model_to_scene(test_pos_src(i,1),amplifier,max_length);
    test_pos_src_in_scene(i,2) = model_to_scene(test_pos_src(i,2),amplifier,max_length);
end

train_pos_robot = dlmread('train_robot_pos.txt');
train_pos_robot = reshape(train_pos_robot,2,length(train_pos_robot)/2).';
len_train_robot = length(train_pos_robot);

test_pos_robot = dlmread('test_robot_pos.txt');
test_pos_robot = reshape(test_pos_robot,2,length(test_pos_robot)/2).';
len_test_robot = length(test_pos_robot);

%number of robot positions in one source position
train_robot_src_ratio = len_train_robot/len_train_src;
test_robot_src_ratio = len_test_robot/len_test_src;
% 


nx = 1000;nz=1000;width=2;
floor_plan = dlmread('Rs_int_vp_1000-1000.dat');floor_plan = reshape(floor_plan,nx,nz).';
floor_plan = -floor_plan*2;
nx = floor(nx/amplifier);
nz = floor(nz/amplifier);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%convert to model domain:
for i =1:nx
    for j =1:nz
        floor_plan(i,j) = floor_plan(1 + 2*(i-1),1+2*(j-1));
    end
end
floor_plan = floor_plan(1:nx,1:nz);

floor_plan_clean = floor_plan;

%robot position in both training and test dataset:
train_pos_robot = [171,253];
floor_plan(train_pos_robot(1)-width:train_pos_robot(1)+width,train_pos_robot(2)-width:train_pos_robot(2)+width) = -1;

%source position in training dataset:
for i = 1:length(train_pos_src)
    floor_plan(train_pos_src(i,1)-width:train_pos_src(i,1)+width,train_pos_src(i,2)-width:train_pos_src(i,2)+width) = 2;
%     for j = 1+(i-1)*train_robot_src_ratio:i*train_robot_src_ratio
%         floor_plan(train_pos_robot(j,1)-width:train_pos_robot(j,1)+width,train_pos_robot(j,2)-width:train_pos_robot(j,2)+width) = 1;
%     end
end

%source position in test dataset:
for i = 1:length(test_pos_src)
    floor_plan(test_pos_src(i,1)-width:test_pos_src(i,1)+width,test_pos_src(i,2)-width:test_pos_src(i,2)+width) = 1;
%     for j = 1+(i-1)*test_robot_src_ratio:i*test_robot_src_ratio
%         floor_plan(test_pos_robot(j,1)-width:test_pos_robot(j,1)+width,test_pos_robot(j,2)-width:test_pos_robot(j,2)+width) = -1;
%     end
    
end

figure(2)
imagesc(floor_plan);caxis([-2 2]);colormap('jet');hold on;
axis equal;
xlim([1 nx]);
ylim([1 nz]);
xlabel('Length (m)');
ylabel('Width (m)');
title('Sampling of locations of robots and source')
set(gca,'XAxisLocation','top');colormap('jet');
set(gca, 'XTick', [1 nx/4 nx*2/4 nx*3/4 nx])            
set(gca,'XTickLabel',{'-5.0','-2.5','0.0','2.5','5.0'}) 
set(gca, 'YTick', [1 nz/4 nz*2/4 nz*3/4 nz])            
set(gca,'YTickLabel',{'-5.0','-2.5','0.0','2.5','5.0'}) 
imwrite(getframe(gcf).cdata, ['test_src_location.png'])


shortest_path_in_scene = dlmread('0.64_1.22to0.94_-2.06shortest_path.txt');
shortest_path_in_scene=reshape(shortest_path_in_scene,2,length(shortest_path_in_scene)/2).';
shortest_path = zeros(length(shortest_path_in_scene),2);
for i = 1:length(shortest_path_in_scene)
    shortest_path(i,1) = scene_to_model(shortest_path_in_scene(i,1),amplifier,max_length);
    shortest_path(i,2) = scene_to_model(shortest_path_in_scene(i,2),amplifier,max_length);
end

floor_plan = floor_plan_clean;
for i = 1:length(shortest_path)
    if i == 1
        floor_plan(shortest_path(i,1)-width:shortest_path(i,1)+width,shortest_path(i,2)-width:shortest_path(i,2)+width) = -1;
    elseif i == length(shortest_path)
        floor_plan(shortest_path(i,1)-width:shortest_path(i,1)+width,shortest_path(i,2)-width:shortest_path(i,2)+width) = 2;
    else
        floor_plan(shortest_path(i,1)-width:shortest_path(i,1)+width,shortest_path(i,2)-width:shortest_path(i,2)+width) = 1;
    end
end

%robot position:126 342 (-2.48,1.84)
%src position: 275   138 (0.50,-2.24)

figure(3)
imagesc(floor_plan);caxis([-2 2]);colormap('jet');hold on;
axis equal;
xlim([1 nx]);
ylim([1 nz]);
xlabel('Length (m)');
ylabel('Width (m)');
title('The shortest path')
set(gca,'XAxisLocation','top');colormap('jet');
set(gca, 'XTick', [1 nx/4 nx*2/4 nx*3/4 nx])            
set(gca,'XTickLabel',{'-5.0','-2.5','0.0','2.5','5.0'}) 
set(gca, 'YTick', [1 nz/4 nz*2/4 nz*3/4 nz])            
set(gca,'YTickLabel',{'-5.0','-2.5','0.0','2.5','5.0'}) 
imwrite(getframe(gcf).cdata, ['shortest_path.png'])


%training results:

export_src_real_list = dlmread('export_src_real_list.txt');
export_src_fake_list = dlmread('export_src_fake_list.txt');

export_src_real_list = reshape(export_src_real_list,2,length(export_src_real_list)/2).';
export_src_fake_list = reshape(export_src_fake_list,2,length(export_src_fake_list)/2).';

export_src_real_list_1d = zeros(1,length(export_src_real_list));
export_src_fake_list_1d = zeros(1,length(export_src_fake_list));

for i = 1:length(export_src_fake_list)
    export_src_real_list_1d(i) = export_src_real_list(i,1)*nz + export_src_real_list(i,2);
    export_src_fake_list_1d(i) = export_src_fake_list(i,1)*nz + export_src_fake_list(i,2);
end


%test results
floor_plan = floor_plan_clean;
test_src_real_list = dlmread('test_src_real_list.txt');
test_src_fake_list = dlmread('test_src_fake_list.txt');

test_src_real_list = reshape(test_src_real_list,2,length(test_src_real_list)/2).';
test_src_fake_list = reshape(test_src_fake_list,2,length(test_src_fake_list)/2).';

test_src_real_list_1d = zeros(1,length(test_src_real_list));
test_src_fake_list_1d = zeros(1,length(test_src_fake_list));

for i = 1:length(test_src_fake_list)
    test_src_real_list_1d(i) = test_src_real_list(i,1)*nz + test_src_real_list(i,2);
    test_src_fake_list_1d(i) = test_src_fake_list(i,1)*nz + test_src_fake_list(i,2);
end

figure(4)
subplot(2,2,1)
plot(export_src_real_list_1d,export_src_real_list_1d,'r');hold on;
plot(export_src_real_list_1d,export_src_fake_list_1d,'k.');hold off;
axis equal;
xlim([48000 160000]);ylim([48000 160000]);
xlabel('Real source location');
ylabel('Estimated source location');
title('Training dataset')
r2 = R2(export_src_real_list_1d,export_src_fake_list_1d);
h=text(48000, 160000,['R^2 = ' num2str(r2,'%8.6f')])



subplot(2,2,2)
plot(test_src_real_list_1d,test_src_real_list_1d,'r');hold on;
plot(test_src_real_list_1d,test_src_fake_list_1d,'k.');hold off;
axis equal;
xlim([48000 160000]);ylim([48000 160000]);
xlabel('Real source location');
ylabel('Estimated source location');
title('Test dataset')
r2 = R2(test_src_real_list_1d,test_src_fake_list_1d);
h=text(48000, 160000,['R^2 = ' num2str(r2,'%8.6f')])

for i = 1:length(src_real_list)/2
    floor_plan(src_real_list(1+(i-1)*2),src_real_list(2+(i-1)*2)) = 2;
    floor_plan(src_fake_list(1+(i-1)*2),src_fake_list(2+(i-1)*2)) = -1;
end

figure(5)
imagesc(floor_plan);caxis([-2 2]);colormap('jet');hold on;
axis equal;
xlim([1 nx]);
ylim([1 nz]);
xlabel('Length (m)');
ylabel('Width (m)');
title('The shortest path')
set(gca,'XAxisLocation','top');colormap('jet');
set(gca, 'XTick', [1 nx/4 nx*2/4 nx*3/4 nx])            
set(gca,'XTickLabel',{'-5.0','-2.5','0.0','2.5','5.0'}) 
set(gca, 'YTick', [1 nz/4 nz*2/4 nz*3/4 nz])            
set(gca,'YTickLabel',{'-5.0','-2.5','0.0','2.5','5.0'}) 
imwrite(getframe(gcf).cdata, ['prediction_error.png'])



