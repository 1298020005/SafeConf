"""Read immutable public HDF5 objects with metered HTTP Ranges and disk cache.

Cached bytes are opaque storage, not evaluation answers. Logical expression
rows are selected by the caller's frozen role registry before materialization.
"""
from __future__ import annotations
import io,json,hashlib,time,urllib.request,fcntl,os,threading,requests
from pathlib import Path
from .submission_evidence import write_json


class MeteredHTTPFile(io.RawIOBase):
    def __init__(self,url,cache_dir,ledger_path,cap=100_000_000_000,reserved=38_000_000_000,block_size=1<<20):
        self.url=url;self.cache=Path(cache_dir);self.cache.mkdir(parents=True,exist_ok=True)
        self.ledger=Path(ledger_path);self.cap=cap;self.reserved=reserved;self.block=block_size;self.position=0
        self.sessions=threading.local()
        with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=30) as response:
            self.size=int(response.headers['Content-Length']);self.etag=response.headers.get('ETag')
        identity={'url':url,'size':self.size,'etag':self.etag,'block_size':block_size}
        p=self.cache/'OBJECT_IDENTITY.json'
        if p.exists() and json.loads(p.read_text())!=identity:raise RuntimeError('remote object identity changed')
        write_json(p,identity)
    def readable(self):return True
    def seekable(self):return True
    def writable(self):return False
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        pos=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if pos<0:raise ValueError('negative seek')
        self.position=pos;return pos
    def _block(self,number):
        path=self.cache/f'{number:08d}.bin'
        begin=number*self.block;end=min(begin+self.block,self.size)-1
        if path.exists():
            if path.stat().st_size!=end-begin+1:raise RuntimeError('incomplete cached Range')
            return path.read_bytes()
        if not hasattr(self.sessions,'session'):self.sessions.session=requests.Session()
        for attempt in range(4):
            try:
                with self.sessions.session.get(self.url,headers={'Range':f'bytes={begin}-{end}'},timeout=(15,45),stream=True) as response:
                    if response.status_code!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {begin}-{end}/'):
                        raise RuntimeError('Range response mismatch')
                    payload=response.content
                if len(payload)!=end-begin+1:raise RuntimeError('Range response incomplete')
                with self.ledger.with_suffix('.lock').open('a') as lock:
                    fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
                    ledger=json.loads(self.ledger.read_text()) if self.ledger.exists() else {'this_run_network_payload_bytes':0}
                    used=ledger['this_run_network_payload_bytes']+len(payload)
                    if self.reserved+used>self.cap:raise RuntimeError('cumulative download allowance reached')
                    ledger['this_run_network_payload_bytes']=used
                    ledger['range_access_payload_bytes']=ledger.get('range_access_payload_bytes',0)+len(payload)
                    write_json(self.ledger,ledger)
                    fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
                tmp=path.with_name(path.name+f'.{os.getpid()}.{threading.get_ident()}.tmp')
                tmp.write_bytes(payload);os.replace(tmp,path)
                return payload
            except RuntimeError:raise
            except Exception:
                if attempt==3:raise
                time.sleep(2*(attempt+1))
        raise RuntimeError('Range unavailable')
    def prefetch_blocks(self,numbers):
        """Fetch a contiguous run with one request, retaining 1MiB cache keys."""
        if not numbers:return
        if any(b!=a+1 for a,b in zip(numbers,numbers[1:])):raise ValueError('contiguous blocks required')
        if all((self.cache/f'{n:08d}.bin').exists() for n in numbers):return
        begin=numbers[0]*self.block;end=min((numbers[-1]+1)*self.block,self.size)-1
        expected=end-begin+1
        if not hasattr(self.sessions,'session'):self.sessions.session=requests.Session()
        for attempt in range(4):
            try:
                with self.ledger.with_suffix('.lock').open('a') as lock:
                    fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
                    ledger=json.loads(self.ledger.read_text()) if self.ledger.exists() else {'this_run_network_payload_bytes':0}
                    if self.reserved+ledger['this_run_network_payload_bytes']+expected+3*8*self.block>self.cap:
                        raise RuntimeError('cumulative download allowance reached')
                with self.sessions.session.get(self.url,headers={'Range':f'bytes={begin}-{end}'},timeout=(15,60),stream=True) as response:
                    if response.status_code!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {begin}-{end}/'):
                        raise RuntimeError('Range response mismatch')
                    payload=response.content
                if len(payload)!=expected:raise RuntimeError('Range response incomplete')
                with self.ledger.with_suffix('.lock').open('a') as lock:
                    fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
                    ledger=json.loads(self.ledger.read_text()) if self.ledger.exists() else {'this_run_network_payload_bytes':0}
                    ledger['this_run_network_payload_bytes']+=len(payload)
                    ledger['range_access_payload_bytes']=ledger.get('range_access_payload_bytes',0)+len(payload)
                    write_json(self.ledger,ledger)
                    if self.reserved+ledger['this_run_network_payload_bytes']>self.cap:raise RuntimeError('cumulative download allowance reached')
                for number in numbers:
                    path=self.cache/f'{number:08d}.bin';part=payload[(number*self.block-begin):min((number+1)*self.block-begin,len(payload))]
                    if path.exists():
                        if path.read_bytes()!=part:raise RuntimeError('immutable cached Range changed')
                        continue
                    tmp=path.with_name(path.name+f'.{os.getpid()}.{threading.get_ident()}.tmp');tmp.write_bytes(part);os.replace(tmp,path)
                return
            except RuntimeError:raise
            except Exception:
                if attempt==3:raise
                time.sleep(2*(attempt+1))
    def read(self,size=-1):
        if size<0:size=self.size-self.position
        size=min(size,self.size-self.position)
        if size<=0:return b''
        start,end=self.position,self.position+size;parts=[]
        for number in range(start//self.block,(end-1)//self.block+1):
            payload=self._block(number);left=max(start-number*self.block,0);right=min(end-number*self.block,len(payload))
            parts.append(payload[left:right])
        self.position=end;return b''.join(parts)
    def readinto(self,buffer):
        payload=self.read(len(buffer));buffer[:len(payload)]=payload;return len(payload)
