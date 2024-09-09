import requests
from shlex import quote
import json, copy, os, time
from requests_toolbelt import MultipartEncoder

class alist():
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, user, pwd, host='http://146.235.33.108:5244'):
        self.host = host
        self.headers = {
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json'
        }

        self.user = user
        self.pwd = pwd

        self.t = time.time()
        self.headers['Authorization'] = self.get_token()

    def request(self, method, url, payload, headers='', **kwargs):
        headers = self.headers if headers == '' else headers
        response = requests.request(method, url, headers=headers, data=payload, **kwargs)
        data = json.loads(response.text)
        if data['message'] == 'success':
            return data['data']
        else:
            raise Exception(f'{data}')

    def check_token(self):
        if time.time() - self.t > 48 * 3600:
            print('update token')
            self.t = time.time()
            self.headers['Authorization'] = self.get_token()

    def get_token(self):
        url = self.host + "/api/auth/login"

        payload = json.dumps({
            "username": self.user,
            "password": self.pwd
        })

        data = self.request("POST", url, payload)
        return data['token']

    def listdir(self, path):
        self.check_token()

        url = self.host + "/api/fs/list"

        payload = json.dumps({
            "path": path,
            "password": "",
            "page": 1,
            "per_page": 0,
            "refresh": False
        })

        data = self.request("POST", url, payload)
        return data['content']

    def info(self, path):
        self.check_token()
        url = self.host + "/api/fs/get"

        payload = json.dumps({
        "path": path,
        "password": "",
        "page": 1,
        "per_page": 0,
        "refresh": False
        })

        data = self.request("POST", url, payload)
        return data

    def download(self, file, dst='./'):
        self.check_token()
        os.makedirs(dst, exist_ok=True)

        data = self.info(file)
        url = data['raw_url']

        response = requests.request("GET", url)
        with open(os.path.join(dst, data['name']), "wb") as out:
            out.write(response.content)

    def remove(self, path):
        """
        删除文件:
            aaa/temp
            /temp

        删除文件夹：
            /aaa
        """
        self.check_token()
        url = self.host + "/api/fs/remove"

        path, file = path.rsplit('/', 1)
        if not path:
            path = '/'

        payload = json.dumps({
        "names": [
        file
        ],
        "dir": path
        })

        return self.request("POST", url, payload)

    def upload(self, path, dst):
        self.check_token()
        url = self.host + "/api/fs/put"

        path = path.replace('\\', '/')
        if '/' in path:
            folder, file = path.rsplit('/', 1)
        else:
            folder, file = '/', path

        _headers = copy.deepcopy(self.headers)
        _headers['Content-Type'] = 'application/octet-stream'
        _headers['Content-Length'] = str(os.path.getsize(path))
        _headers['File-Path'] = quote(str(os.path.join(dst, file)).replace('\\', '/'))
        _headers['As-Task'] = 'true'

        with open(path, 'rb') as f:
            response = requests.put(url, headers=_headers, data=f)

        # 检查响应
        if response.status_code == 200:
            print(f"Upload successful: {path}")
        else:
            raise Exception(f'{response}')

    def mkdir(self, folder):
        self.check_token()
        url = self.host + "/api/fs/mkdir"

        payload = json.dumps({
        "path": folder
        })

        data = self.request("POST", url, payload)
        return data
        


if __name__ == '__main__':
    a = alist('admin', '***')

    # listdir
    cur_files = [i['name'] for i in a.listdir('/')]
    print(cur_files)

    # mkdir
    if 'test' not in cur_files:
        print(f'mkdir: /test')
        a.mkdir('/test')
    
    # # upload 会覆盖同名文件
    a.upload('alist.py', '/')
    a.upload(r'D:\code\dl_helper\dl_helper\alist.py', '/test')

    # download
    a.download('/test/alist.py', dst=r'temp/')

    # remove
    a.remove('/test/alist.py')