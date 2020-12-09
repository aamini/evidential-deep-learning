clear;

dataset_path = '/data/nyu_depth';
output_path = '/data/nyu_depth_processed';
skip_n = 1;

addpath(genpath('~/Documents/MATLAB'));

places = dir(dataset_path);
places = places(3:end);
for i=1:length(places)
    place = places(i).name;

    places_done = dir(output_path);
    done = false;
    for j=1:length(places_done)
        if strcmp(place, places_done(j).name)
            done = true;
            break
        end
    end

    if done
        fprintf('skipping %s \n',place);
         continue
    end
    mkdir(strcat(output_path, '/', place));
    sync = get_synched_frames(strcat(dataset_path, '/', place));
    fprintf('preprocessing %d in %s \n', int16(length(sync)/skip_n), place);
    for j=1:skip_n:length(sync)
        try
          rgb = imread(strcat(dataset_path, '/', place, '/', sync(j).rawRgbFilename));
          depth = imread(strcat(dataset_path, '/', place, '/', sync(j).rawDepthFilename));
          depth_proj = project_depth_map(swapbytes(depth), rgb);
          depth_fill = fill_depth_colorization(double(rgb)/255,depth_proj,0.9);

          disp = (1./depth_fill) * 255.;
          disp = uint8(max(0, min(255, disp)));
        catch
          continue;
        end

        save(strcat(output_path, '/', place, '/scan_', num2str(j)), 'rgb','disp');

        fprintf('.');
    end
    fprintf('\n');

end
