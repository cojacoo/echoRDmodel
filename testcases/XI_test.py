mcinif='mcini_XI'
mcpick='XI.pickle3'
runname='testXI'
pathdir='../echoRD'
wdir='.'
legacy_pick=True

import sys
sys.path.append(pathdir)

import run_echoRD as rE
rE.echoRD_job(mcinif=mcinif,mcpick=mcpick,runname=runname,wdir=wdir,pathdir=pathdir,legacy_pick=legacy_pick,model_restart=True)
