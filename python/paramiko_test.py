import paramiko

ssh = paramiko.SSHClient()
ssh.connect('127.0.0.1')
print(ssh.exec_command('ls -l'))
