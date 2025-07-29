def time_to_frame(time_str):
    h, m, s = map(float, time_str.split(':'))
    total_seconds = int(h * 3600 + m * 60 + s)
    return int(total_seconds * 120)

time_strs = [
"00:01:25","00:01:27",
"00:01:27","00:01:28",
"00:01:40","00:01:42",
"00:01:43","00:01:44",
"00:01:56","00:01:58",
"00:01:58","00:02:05",
"00:02:15","00:02:17",
"00:02:18","00:02:20",]

frame_indices = [time_to_frame(ts) for ts in time_strs]
i = 0
for i in range(0,len(frame_indices), 2):
    start_frame = frame_indices[i]
    end_frame = frame_indices[i + 1]
    print(f"{start_frame},{end_frame}")