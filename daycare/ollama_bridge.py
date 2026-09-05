#!/usr/bin/env python3
"""Tiny Ollama-compatible HTTP bridge for the persisted Metatron champion."""
from __future__ import annotations
import argparse,json,pickle,sys
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path

class Handler(BaseHTTPRequestHandler):
    model=None
    def _send(self,code,obj):
        b=json.dumps(obj).encode(); self.send_response(code); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path=='/api/tags': return self._send(200,{'models':[{'name':'metatron:champion','model':'metatron:champion','size':0,'digest':'local-champion'}]})
        return self._send(404,{'error':'not found'})
    def do_POST(self):
        n=int(self.headers.get('Content-Length','0')); body=json.loads(self.rfile.read(n) or '{}')
        prompt=body.get('prompt','')
        if body.get('messages'): prompt='\n'.join(str(x.get('content','')) for x in body['messages'] if x.get('role')!='system')
        opts=body.get('options') or {}
        text=self.model.generate(prompt,max_new=int(opts.get('num_predict',96)),temperature=float(opts.get('temperature',0.7)))
        if self.path in ('/api/generate','/api/chat'):
            out={'model':'metatron:champion','created_at':'','done':True,'response':text}
            if self.path.endswith('/chat'): out['message']={'role':'assistant','content':text}
            return self._send(200,out)
        return self._send(404,{'error':'not found'})
    def log_message(self,*a): pass

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default='daycare_state/champion/model.pkl'); ap.add_argument('--host',default='0.0.0.0'); ap.add_argument('--port',type=int,default=11435); a=ap.parse_args(); p=Path(a.checkpoint)
    if not p.is_file(): raise SystemExit('champion checkpoint missing: '+str(p))
    root=p.parents[2] if p.parts[-2]=='champion' else Path('.')
    candidates=list(root.rglob('metatron_v2.py'))
    if candidates:
        import importlib.util; name='metatron_v2_real'; sp=importlib.util.spec_from_file_location(name,candidates[0]); m=importlib.util.module_from_spec(sp); sys.modules[name]=m; sp.loader.exec_module(m)
    with open(p,'rb') as f: Handler.model=pickle.load(f)
    ThreadingHTTPServer((a.host,a.port),Handler).serve_forever()
if __name__=='__main__': main()
