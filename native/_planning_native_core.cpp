// Windows resource ownership only; no numerical kernels or Python callbacks.
#ifndef Py_LIMITED_API
#define Py_LIMITED_API 0x030B0000
#endif
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#ifdef Py_GIL_DISABLED
#error "planning resource helper requires the CPython GIL build"
#endif
#ifndef _WIN32
#error "planning resource helper requires Windows"
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <vector>
#include <memory>
#include <string>
#include <new>

namespace {
constexpr const char* capsule_name = "anygeometry.planning.owner.v1";
struct Slot { HANDLE value=nullptr; bool poisoned=false, closing=false; DWORD error=0; unsigned borrowers=0; };
struct Child { Slot process, thread, io[3]; DWORD pid=0, tid=0; bool attempted=false; };
struct Owner {
    SRWLOCK lock=SRWLOCK_INIT;
    CONDITION_VARIABLE changed=CONDITION_VARIABLE_INIT;
    Slot job;
    std::vector<std::unique_ptr<Child>> children;
    std::vector<std::unique_ptr<Slot>> observations;
    bool poisoned=false, retiring=false, job_attempted=false;
    // Deterministic pre-operation failure seam; never invokes Python in a transition.
    std::string fault;
    DWORD fault_error=5;
    bool fault_after=false;
    Owner* quarantine_next=nullptr;
};
SRWLOCK quarantine_lock=SRWLOCK_INIT;
Owner* quarantine=nullptr; // Uncertain slots remain tracked until process retirement.
struct Guard { Owner* o; explicit Guard(Owner* p):o(p){AcquireSRWLockExclusive(&o->lock);} ~Guard(){ReleaseSRWLockExclusive(&o->lock);} };
Owner* owner(PyObject* p){return static_cast<Owner*>(PyCapsule_GetPointer(p,capsule_name));}
PyObject* error(const char* op,DWORD code){
    PyErr_SetFromWindowsErr(static_cast<int>(code));
    PyObject *type=nullptr,*value=nullptr,*tb=nullptr;
    PyErr_Fetch(&type,&value,&tb); PyErr_NormalizeException(&type,&value,&tb);
    if(value){PyObject* name=PyUnicode_FromString(op);if(name){PyObject_SetAttrString(value,"job_operation",name);Py_DECREF(name);}}
    PyErr_Restore(type,value,tb);return nullptr;
}
bool fault(Owner* o,const char* op,bool after=false){
    if(o->fault==op && o->fault_after==after){SetLastError(o->fault_error);o->fault.clear();return true;}return false;
}
bool close_slot(Owner* o,Slot& s){
    while(s.closing)SleepConditionVariableSRW(&o->changed,&o->lock,INFINITE,0);
    if(!s.value)return true;
    if(s.poisoned){SetLastError(s.error);return false;}
    s.closing=true;
    while(s.borrowers)SleepConditionVariableSRW(&o->changed,&o->lock,INFINITE,0);
    BOOL ok=!fault(o,"CloseHandle") && CloseHandle(s.value);
    if(ok && fault(o,"CloseHandle",true))ok=FALSE;
    DWORD code=ok?0:GetLastError();
    s.closing=false;
    if(ok){s.value=nullptr;s.error=0;}
    else{s.poisoned=true;s.error=code;o->poisoned=true;}
    WakeAllConditionVariable(&o->changed);
    SetLastError(code);return ok!=FALSE;
}
Slot* find(Owner* o,HANDLE h){
    if(h && o->job.value==h)return &o->job;
    for(auto& c:o->children){
        if(h && c->process.value==h)return &c->process;
        if(h && c->thread.value==h)return &c->thread;
        for(auto& io:c->io)if(h && io.value==h)return &io;
    }
    for(auto& s:o->observations)if(h && s->value==h)return s.get();
    return nullptr;
}
bool usable(Slot* s){if(!s || !s->value || s->closing){SetLastError(ERROR_INVALID_HANDLE);return false;}if(s->poisoned){SetLastError(s->error);return false;}return true;}
void destroy(PyObject* capsule){
    Owner* o=owner(capsule);if(!o){PyErr_Clear();return;}
    bool uncertain=false;
    // Other C calls retain the capsule through their argument tuple. No borrowers
    // can survive the final capsule decref; explicit close uses the same slots.
    {Guard g(o);
        if(o->job.value && !o->job.poisoned)TerminateJobObject(o->job.value,1);
        for(auto& c:o->children){for(auto& io:c->io)uncertain|=!close_slot(o,io);uncertain|=!close_slot(o,c->thread);uncertain|=!close_slot(o,c->process);}
        for(auto& s:o->observations)uncertain|=!close_slot(o,*s);
        uncertain|=!close_slot(o,o->job);
    }
    if(uncertain){AcquireSRWLockExclusive(&quarantine_lock);o->quarantine_next=quarantine;quarantine=o;ReleaseSRWLockExclusive(&quarantine_lock);}else delete o;
}
PyObject* new_owner(PyObject*,PyObject*){
    Owner* o=new(std::nothrow) Owner();if(!o)return PyErr_NoMemory();
    PyObject* p=PyCapsule_New(o,capsule_name,destroy);if(!p)delete o;return p;
}
PyObject* reserve_child(PyObject*,PyObject* p){
    Owner* o=owner(p);if(!o)return nullptr;
    size_t n=0;DWORD code=0;
    try {Guard g(o);if(o->poisoned||o->retiring||!usable(&o->job))code=ERROR_INVALID_HANDLE;
        else{n=o->children.size();o->children.emplace_back(std::make_unique<Child>());}}
    catch(const std::bad_alloc&){return PyErr_NoMemory();}
    if(code)return error("reserve child on unusable owner",code);
    return PyLong_FromSize_t(n);
}
PyObject* job_create(PyObject*,PyObject* p){
    Owner* o=owner(p);if(!o)return nullptr;DWORD code=0;HANDLE h=nullptr;
    {Guard g(o);if(o->poisoned||o->retiring||o->job_attempted)code=ERROR_ALREADY_EXISTS;else {o->job_attempted=true;o->job.value=CreateJobObjectW(nullptr,nullptr);h=o->job.value;if(!h)code=GetLastError();}}
    if(code)return error("CreateJobObjectW",code);return PyLong_FromVoidPtr(h);
}
PyObject* state(PyObject*,PyObject* args){
    PyObject* p;Py_ssize_t index=-1;if(!PyArg_ParseTuple(args,"O|n",&p,&index))return nullptr;
    Owner* o=owner(p);if(!o)return nullptr;
    HANDLE job,process=nullptr,thread=nullptr,io[3]={};DWORD pid=0,tid=0;bool poisoned=false;unsigned borrowers=0;
    {Guard g(o);job=o->job.value;poisoned=o->poisoned;
        if(index>=0){if(static_cast<size_t>(index)>=o->children.size()){PyErr_SetString(PyExc_IndexError,"child slot");return nullptr;}
            auto& c=*o->children[index];process=c.process.value;thread=c.thread.value;pid=c.pid;tid=c.tid;
            borrowers=c.process.borrowers;poisoned|=c.process.poisoned||c.thread.poisoned;
            for(int i=0;i<3;++i){io[i]=c.io[i].value;poisoned|=c.io[i].poisoned;}}
    }
    return Py_BuildValue("{s:K,s:K,s:K,s:k,s:k,s:(KKK),s:O,s:k}","job",(unsigned long long)job,"process",(unsigned long long)process,"thread",(unsigned long long)thread,"pid",(unsigned long)pid,"tid",(unsigned long)tid,"io",(unsigned long long)io[0],(unsigned long long)io[1],(unsigned long long)io[2],"poisoned",poisoned?Py_True:Py_False,"borrowers",(unsigned long)borrowers);
}
PyObject* close_handle(PyObject*,PyObject* args){
    PyObject* p;unsigned long long raw;Py_ssize_t index=-2;int kind=0;
    if(!PyArg_ParseTuple(args,"OK|ni",&p,&raw,&index,&kind))return nullptr;
    Owner* o=owner(p);if(!o)return nullptr;DWORD code=0;
    // Never hold the GIL while waiting for a borrower to finish. A borrower
    // decrements its count before restoring its Python thread state.
    Py_BEGIN_ALLOW_THREADS
    {Guard g(o);Slot* s=nullptr;
        if(index==-1)s=&o->job;
        else if(index>=0 && kind>=0 && kind<5 && (size_t)index<o->children.size()){
            Child& c=*o->children[index];s=kind==0?&c.process:kind==1?&c.thread:&c.io[kind-2];
        }else if(index>=0 && kind==5 && (size_t)index<o->observations.size())s=o->observations[index].get();
        else if(index==-2)s=find(o,(HANDLE)raw);
        if(!s || (s->value && s->value!=(HANDLE)raw))code=ERROR_INVALID_HANDLE;
        else {
            if(s==&o->job){o->retiring=true;for(auto& observation:o->observations)if(!close_slot(o,*observation)&&!code)code=GetLastError();}
            if(!close_slot(o,*s)&&!code)code=GetLastError();
        }
    }
    Py_END_ALLOW_THREADS
    if(code)return error("CloseHandle",code);Py_RETURN_NONE;
}
PyObject* query(PyObject*,PyObject* args){
    PyObject* p;int kind;unsigned long size;
    if(!PyArg_ParseTuple(args,"Oik",&p,&kind,&size))return nullptr;
    Owner* o=owner(p);if(!o)return nullptr;
    if(size>1024*1024){PyErr_SetString(PyExc_ValueError,"query buffer exceeds cap");return nullptr;}
    std::vector<char> data;try{data.resize(size);}catch(const std::bad_alloc&){return PyErr_NoMemory();}
    DWORD code=0;{Guard g(o);if(!usable(&o->job)||!QueryInformationJobObject(o->job.value,(JOBOBJECTINFOCLASS)kind,data.data(),size,nullptr))code=GetLastError();}
    if(code)return error("QueryInformationJobObject",code);return PyBytes_FromStringAndSize(data.data(),size);
}
PyObject* configure(PyObject*,PyObject* args){
    PyObject* p;int kind;const char* data;Py_ssize_t size;
    if(!PyArg_ParseTuple(args,"Oiy#",&p,&kind,&data,&size))return nullptr;
    Owner* o=owner(p);if(!o)return nullptr;DWORD code=0;
    {Guard g(o);if(!usable(&o->job)||!SetInformationJobObject(o->job.value,(JOBOBJECTINFOCLASS)kind,(void*)data,(DWORD)size))code=GetLastError();}
    if(code)return error("SetInformationJobObject",code);Py_RETURN_NONE;
}
PyObject* terminate(PyObject*,PyObject* p){
    Owner* o=owner(p);if(!o)return nullptr;DWORD code=0;
    {Guard g(o);if(!usable(&o->job)||!TerminateJobObject(o->job.value,1))code=GetLastError();}
    if(code)return error("TerminateJobObject",code);Py_RETURN_NONE;
}
PyObject* assign(PyObject*,PyObject* args){
    PyObject* p;unsigned long long raw;if(!PyArg_ParseTuple(args,"OK",&p,&raw))return nullptr;
    Owner* o=owner(p);if(!o)return nullptr;DWORD code=0;
    {Guard g(o);if(o->poisoned||o->retiring)code=ERROR_INVALID_HANDLE;
        else if(!usable(&o->job)||!AssignProcessToJobObject(o->job.value,(HANDLE)raw))code=GetLastError();}
    if(code)return error("AssignProcessToJobObject",code);Py_RETURN_NONE;
}
PyObject* observe(PyObject*,PyObject* args){
    PyObject* p;unsigned long pid;if(!PyArg_ParseTuple(args,"Ok",&p,&pid))return nullptr;
    Owner* o=owner(p);if(!o)return nullptr;DWORD code=0;size_t index=0;HANDLE handle=nullptr;
    try{Guard g(o);
        if(o->poisoned||o->retiring||!usable(&o->job))code=ERROR_INVALID_HANDLE;
        else {
            index=o->observations.size();o->observations.emplace_back(std::make_unique<Slot>());
            Slot& slot=*o->observations[index];slot.value=OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION,FALSE,pid);
            if(!slot.value)code=GetLastError();
            else {BOOL member=FALSE;
                if(!IsProcessInJob(slot.value,o->job.value,&member)||!member){code=ERROR_ACCESS_DENIED;close_slot(o,slot);}
                else handle=slot.value;
            }
        }
    }catch(const std::bad_alloc&){return PyErr_NoMemory();}
    if(code)return error("OpenProcess owned member",code);
    return Py_BuildValue("Kn",(unsigned long long)handle,(Py_ssize_t)index);
}
struct Wide {
    wchar_t* p=nullptr;Py_ssize_t size=0;
    ~Wide(){PyMem_Free(p);}
    bool get(PyObject* value,bool block=false){
        p=PyUnicode_AsWideCharString(value,&size);
        if(!p)return false;
        if(!block && static_cast<Py_ssize_t>(wcslen(p))!=size){PyErr_SetString(PyExc_ValueError,"embedded null character");return false;}
        if(block && (size<2 || p[size-1]!=0 || p[size-2]!=0)){PyErr_SetString(PyExc_ValueError,"unterminated environment block");return false;}
        return true;
    }
};
struct Attributes {LPPROC_THREAD_ATTRIBUTE_LIST p=nullptr;bool initialized=false;~Attributes(){DWORD code=GetLastError();if(initialized)DeleteProcThreadAttributeList(p);if(p)HeapFree(GetProcessHeap(),0,p);SetLastError(code);}};
PyObject* create(PyObject*,PyObject* args){
    PyObject *p,*exe,*cmd,*env,*cwd,*out,*err;Py_ssize_t index;
    if(!PyArg_ParseTuple(args,"OnOOOOOO",&p,&index,&exe,&cmd,&env,&cwd,&out,&err))return nullptr;
    Owner* o=owner(p);if(!o)return nullptr;Wide e,c,v,d,f,g;
    if(!e.get(exe)||!c.get(cmd)||!v.get(env,true)||!d.get(cwd)||!f.get(out)||!g.get(err))return nullptr;
    DWORD code=0;const char* operation="CreateProcessW";
    {Guard guard(o);
        if(index<0 || (size_t)index>=o->children.size()){code=ERROR_INVALID_PARAMETER;}
        else if(o->poisoned||o->retiring||!usable(&o->job)){code=ERROR_INVALID_HANDLE;}
        else {
            Child& child=*o->children[index];
            if(child.attempted){code=ERROR_ALREADY_EXISTS;}
            else {
                child.attempted=true;
                auto transaction=[&]()->bool {
                    SECURITY_ATTRIBUTES security={sizeof(SECURITY_ATTRIBUTES),nullptr,TRUE};
                    const wchar_t* paths[]={L"NUL",f.p,g.p};
                    for(int i=0;i<3;++i){
                        operation=i==0?"CreateFileW stdin":i==1?"CreateFileW stdout":"CreateFileW stderr";
                        if(fault(o,operation))return false;
                        HANDLE value=CreateFileW(paths[i],i==0?GENERIC_READ:GENERIC_WRITE,FILE_SHARE_READ|FILE_SHARE_WRITE,&security,i==0?OPEN_EXISTING:CREATE_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
                        if(value==INVALID_HANDLE_VALUE)return false;
                        child.io[i].value=value; // Commit before any Python API or signal.
                    }
                    Attributes attrs;SIZE_T size=0;operation="InitializeProcThreadAttributeList size";
                    if(fault(o,operation))return false;
                    BOOL sized=InitializeProcThreadAttributeList(nullptr,2,0,&size);
                    if(sized || GetLastError()!=ERROR_INSUFFICIENT_BUFFER || !size || size>1024*1024){SetLastError(ERROR_INVALID_DATA);return false;}
                    attrs.p=(LPPROC_THREAD_ATTRIBUTE_LIST)HeapAlloc(GetProcessHeap(),0,size);
                    if(!attrs.p){SetLastError(ERROR_NOT_ENOUGH_MEMORY);return false;}
                    operation="InitializeProcThreadAttributeList";
                    if(fault(o,operation)||!InitializeProcThreadAttributeList(attrs.p,2,0,&size))return false;
                    attrs.initialized=true;
                    operation="UpdateProcThreadAttribute job";
                    if(fault(o,operation)||!UpdateProcThreadAttribute(attrs.p,0,PROC_THREAD_ATTRIBUTE_JOB_LIST,&o->job.value,sizeof(HANDLE),nullptr,nullptr))return false;
                    HANDLE handles[]={child.io[0].value,child.io[1].value,child.io[2].value};
                    operation="UpdateProcThreadAttribute handles";
                    if(fault(o,operation)||!UpdateProcThreadAttribute(attrs.p,0,PROC_THREAD_ATTRIBUTE_HANDLE_LIST,handles,sizeof(handles),nullptr,nullptr))return false;
                    STARTUPINFOEXW startup={};startup.StartupInfo.cb=sizeof(startup);startup.StartupInfo.dwFlags=STARTF_USESTDHANDLES;
                    startup.StartupInfo.hStdInput=handles[0];startup.StartupInfo.hStdOutput=handles[1];startup.StartupInfo.hStdError=handles[2];startup.lpAttributeList=attrs.p;
                    PROCESS_INFORMATION info={};operation="CreateProcessW";
                    if(fault(o,operation))return false;
                    BOOL ok=CreateProcessW(e.p,c.p,nullptr,nullptr,TRUE,EXTENDED_STARTUPINFO_PRESENT|CREATE_NO_WINDOW|CREATE_UNICODE_ENVIRONMENT|CREATE_SUSPENDED,v.p,d.p,&startup.StartupInfo,&info);
                    DWORD last=GetLastError();
                    child.process.value=info.hProcess;child.thread.value=info.hThread;child.pid=info.dwProcessId;child.tid=info.dwThreadId;
                    SetLastError(last);return ok!=FALSE;
                };
                if(!transaction())code=GetLastError();
            }
        }
    }
    if(code)return error(operation,code);Py_RETURN_NONE;
}
PyObject* member(PyObject*,PyObject* args){
    PyObject* p;unsigned long long raw,job;if(!PyArg_ParseTuple(args,"OKK",&p,&raw,&job))return nullptr;
    Owner* o=owner(p);if(!o)return nullptr;DWORD code=0;BOOL result=FALSE;
    {Guard g(o);Slot* s=find(o,(HANDLE)raw);if((HANDLE)job!=o->job.value||!usable(&o->job)||!usable(s))code=ERROR_INVALID_HANDLE;else if(!IsProcessInJob(s->value,o->job.value,&result))code=GetLastError();}
    if(code)return error("IsProcessInJob",code);return PyBool_FromLong(result);
}
PyObject* resume(PyObject*,PyObject* args){
    PyObject* p;unsigned long long raw;Py_ssize_t index=-1;if(!PyArg_ParseTuple(args,"OK|n",&p,&raw,&index))return nullptr;
    Owner* o=owner(p);if(!o)return nullptr;DWORD code=0,result=0xffffffff;
    {Guard g(o);Slot* s=index<0?find(o,(HANDLE)raw):(size_t)index<o->children.size()?&o->children[index]->thread:nullptr;
        if(o->poisoned||o->retiring||(s && s->value!=(HANDLE)raw))code=ERROR_INVALID_HANDLE;
        else if(!usable(s))code=GetLastError();else{result=ResumeThread(s->value);if(result==0xffffffff)code=GetLastError();}}
    if(code)return error("ResumeThread",code);return PyLong_FromUnsignedLong(result);
}
PyObject* wait_process(PyObject*,PyObject* args){
    PyObject* p;Py_ssize_t index;unsigned long milliseconds;
    if(!PyArg_ParseTuple(args,"Onk",&p,&index,&milliseconds))return nullptr;
    Owner* o=owner(p);if(!o)return nullptr;DWORD code=0,result=WAIT_FAILED,exitcode=0;Slot* s=nullptr;
    HANDLE handle=nullptr;
    {Guard g(o);if(index>=0 && (size_t)index<o->children.size())s=&o->children[index]->process;
        if(!usable(s))code=GetLastError();else{++s->borrowers;handle=s->value;}}
    if(code)return error("WaitForSingleObject",code);
    Py_BEGIN_ALLOW_THREADS
    result=WaitForSingleObject(handle,milliseconds);
    if(result==WAIT_FAILED)code=GetLastError();
    else if(result==WAIT_OBJECT_0 && !GetExitCodeProcess(handle,&exitcode))code=GetLastError();
    {Guard g(o);--s->borrowers;WakeAllConditionVariable(&o->changed);}
    Py_END_ALLOW_THREADS
    // Borrow is already released even if this delivers KeyboardInterrupt.
    if(PyErr_CheckSignals()<0)return nullptr;
    if(code)return error("WaitForSingleObject/GetExitCodeProcess",code);
    return Py_BuildValue("kk",(unsigned long)result,(unsigned long)exitcode);
}
PyObject* exit_code(PyObject*,PyObject* args){
    PyObject* p;unsigned long long raw;if(!PyArg_ParseTuple(args,"OK",&p,&raw))return nullptr;
    Owner* o=owner(p);if(!o)return nullptr;DWORD code=0,value=0;
    {Guard g(o);Slot* s=find(o,(HANDLE)raw);if(!usable(s)||!GetExitCodeProcess(s->value,&value))code=GetLastError();}
    if(code)return error("GetExitCodeProcess",code);return PyLong_FromUnsignedLong(value);
}
PyObject* inject(PyObject*,PyObject* args){
    PyObject* p;const char* name;unsigned long code=5;int after=0;
    if(!PyArg_ParseTuple(args,"Os|kp",&p,&name,&code,&after))return nullptr;
    Owner* o=owner(p);if(!o)return nullptr;
    try{Guard g(o);o->fault=name;o->fault_error=code;o->fault_after=after!=0;}catch(const std::bad_alloc&){return PyErr_NoMemory();}
    Py_RETURN_NONE;
}
PyObject* identity(PyObject*,PyObject*){return Py_BuildValue("{s:s,s:i,s:s}","abi","cp311-abi3","resource_abi",1,"build","planning-owner-v1;limited=0x030B0000;gil=required;compiler=" Py_STRINGIFY(_MSC_VER));}
PyMethodDef methods[]={
 {"new_owner",new_owner,METH_NOARGS,nullptr},{"reserve_child",reserve_child,METH_O,nullptr},
 {"job_create",job_create,METH_O,nullptr},{"state",state,METH_VARARGS,nullptr},
 {"close",close_handle,METH_VARARGS,nullptr},{"query",query,METH_VARARGS,nullptr},
 {"configure",configure,METH_VARARGS,nullptr},{"terminate",terminate,METH_O,nullptr},
 {"assign",assign,METH_VARARGS,nullptr},
 {"observe",observe,METH_VARARGS,nullptr},
 {"create",create,METH_VARARGS,nullptr},{"member",member,METH_VARARGS,nullptr},
 {"resume",resume,METH_VARARGS,nullptr},{"wait",wait_process,METH_VARARGS,nullptr},
 {"exit_code",exit_code,METH_VARARGS,nullptr},{"inject_fault",inject,METH_VARARGS,nullptr},
 {"identity",identity,METH_NOARGS,nullptr},{nullptr,nullptr,0,nullptr}};
PyModuleDef module={PyModuleDef_HEAD_INIT,"_planning_native_core",nullptr,-1,methods};
}
PyMODINIT_FUNC PyInit__planning_native_core(){return PyModule_Create(&module);}

