python generate_stage1.py --cuda --song_path $song_path --experiment_name block_placement_ddc2 --checkpoint 130000 --bpm 128 --peak_threshold ${peak_threshold} --temperature 1.0





































python generate_stage2.py --cuda --song_path $song_path --json_file $json_file --experiment_name block_selection_new2 --checkpoint 2150000 --bpm 128 --temperature 1.00 --use_beam_search



[console]::beep(800,700)
[console]::beep(700,700)
[console]::beep(600,700)
