# coach + slot type regularization

# PLAYMUSIC
python test.py --tgt_dm PlayMusic --n_samples 0 --test_mode seen_unseen
python test.py --tgt_dm PlayMusic --n_samples 50 --test_mode seen_unseen

# ADDTOPLAYLIST
python test.py --tgt_dm AddToPlaylist --n_samples 0 --test_mode seen_unseen
python test.py --tgt_dm AddToPlaylist --n_samples 50 --test_mode seen_unseen

# BOOKRESTAURANT
python test.py --tgt_dm BookRestaurant --n_samples 0 --test_mode seen_unseen
python test.py --tgt_dm BookRestaurant --n_samples 50 --test_mode seen_unseen

# GETWEATHER
python test.py --tgt_dm GetWeather --n_samples 0 --test_mode seen_unseen
python test.py --tgt_dm GetWeather --n_samples 50 --test_mode seen_unseen

# RATEBOOK
python test.py --tgt_dm RateBook --n_samples 0 --test_mode seen_unseen
python test.py --tgt_dm RateBook --n_samples 50 --test_mode seen_unseen

# SEARCHCREATEWORK
python test.py --tgt_dm SearchCreativeWork --n_samples 0 --test_mode seen_unseen
python test.py --tgt_dm SearchCreativeWork --n_samples 50 --test_mode seen_unseen

# FINDSCREENINGEVENT
python test.py --tgt_dm SearchScreeningEvent --n_samples 0 --test_mode seen_unseen
python test.py --tgt_dm SearchScreeningEvent --n_samples 50 --test_mode seen_unseen
