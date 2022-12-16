function [model_output] = scene_to_model(scene_input,amplifier,max_length)
    model_output=(scene_input+max_length)*100/amplifier;
    
end

