%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                     The University of Texas at Dallas
%                              Yulang Wu
%                             2022.12.13
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Convert the image to velocity model (dummy)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
pack
close all
clc

output_folder = 'velocity_no_door_no_furnitures';
input_folder = 'scenes';
system(['mkdir ' output_folder])
folder_name = dir(input_folder)
figure(1)
z_dim = zeros(length(folder_name)-4);
for k = 3:length(folder_name)+2
    input_dir = [input_folder '/' folder_name(k).name '/layout/'];
    %no furnitures or doors:
    %floor_plan = imread([input_dir 'floor_trav_no_obj_0','.png']);
    %no doors:
    floor_plan = imread([input_dir 'floor_trav_no_obj_0','.png']);
    
    [nz nx]=size(floor_plan)
    vp = ones(nz,nx);
    for i = 1:nz
        for j = 1:nx
            if floor_plan(i,j) == 255
            vp(i,j) = 0;
            end
        end
    end
 
    fid=fopen([output_folder '/' folder_name(k).name '_vp_' num2str(nz)... 
        '-' num2str(nx) '.dat'],'wt');
    fprintf(fid,'%2.0f',vp);
    fclose(fid);
    
    vp = dlmread([output_folder '/' folder_name(k).name '_vp_' num2str(nz)... 
        '-' num2str(nx) '.dat']);
    vp = reshape(vp, nz, nx);
    subplot(3,5,k-2)
    imagesc(vp)
    title(folder_name(k).name);
    drawnow;
end


