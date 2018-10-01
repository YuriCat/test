import fcntl

ifp = open("hoge.txt", "r")
fcntl.flock(ifp.fileno(), fcntl.LOCK_EX)        # 排他ロック獲得 
# fcntl.flock(ifp.fileno(), fcntl.LOCK_SH)　　　 # 共有ロック獲得 
fcntl.flock(ifp.fileno(), fcntl.LOCK_UN)        # 排他ロック解放 
ifp.close()