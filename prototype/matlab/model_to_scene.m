function [scene_output] = model_to_scene(model_input,amplifier,max_length)
    scene_output=model_input*amplifier/100-max_length;
end

