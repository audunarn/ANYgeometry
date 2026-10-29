"""Independent curved proof verifier (CH3d).

The only numerical imports are verifier-owned modules. Preparation, query
execution, producer roots, and execution/core metadata are never used as an
oracle. Identity utilities and public value classes are shared protocols.
"""
from fractions import Fraction as F
from decimal import Decimal
import hashlib,json,math,platform,importlib.metadata,tomllib,re
from pathlib import Path
from collections.abc import Mapping
from dataclasses import replace
from ._errors import CertifiedHydrostaticsRefusal,RefusalCategory
from ._work import WorkBudget,Meter
from ._ch3_verify_math import Arithmetic,Interval,fail,_active
from ._ch3_verify_geometry import Reconstruction,Computation,pair

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(x):
    data=canonical(x);state=_active.get()
    if state is not None:state[0].unit('verify_serialized_bytes',max(1,(len(data)+65535)//65536))
    return hashlib.sha256(data).hexdigest()
def structural(a,b):
    if type(a) is not type(b):return False
    if type(a) is dict:return a.keys()==b.keys() and all(structural(a[k],b[k]) for k in a)
    if type(a) is list:return len(a)==len(b) and all(structural(x,y) for x,y in zip(a,b))
    return a==b
def rational(x):
    if type(x) is not list or len(x)!=2 or any(type(v) is not int for v in x) or x[1]<=0:fail('invalid rational pair','MALFORMED_PROOF')
    v=F(*x)
    if pair(v)!=x:fail('unnormalized rational pair','MALFORMED_PROOF')
    return v
def plain(x,meter):
    meter.unit('verify_input_element')
    if x is None or type(x) in (str,bool,int):
        if type(x) is int:meter.fit(x)
        return x
    if type(x) in (list,tuple):return [plain(v,meter) for v in x]
    if type(x) is dict:
        if any(type(k) is not str for k in x):fail('non-string object key','MALFORMED_PROOF')
        return {k:plain(v,meter) for k,v in x.items()}
    fail('non-builtin proof value','MALFORMED_PROOF')
def no_duplicates(items):
    d={}
    for k,v in items:
        if k in d:fail('duplicate proof field','MALFORMED_PROOF')
        d[k]=v
    return d
def mismatch(condition,reason,message):
    if not condition:fail(message,'VERIFICATION_FAILED',reason)

def copy_quantity(q):
    from ._forms import QuantityResult
    # Rebuild from numeric public fields; reject arbitrary object coercion.
    if type(q) is not QuantityResult:fail('unexpected quantity type','MALFORMED_PROOF')
    if type(q.name) is not str:fail('quantity name type','MALFORMED_PROOF')
    if q.binary64 is not None and (type(q.binary64) is not tuple or len(q.binary64)!=2 or any(type(x) is not float for x in q.binary64)):fail('binary64 field type','MALFORMED_PROOF')
    if q.decimal is not None and (type(q.decimal) is not tuple or len(q.decimal)!=3 or any(type(x) is not str for x in q.decimal[:2]) or type(q.decimal[2]) is not F):fail('decimal field type','MALFORMED_PROOF')
    for x in (q.numerator,q.norm_squared,q.accuracy):
        if x is not None and type(x) is not F:fail('quantity exact field type','MALFORMED_PROOF')
    enclosure=None
    if q.enclosure is not None:
        if type(q.enclosure) is not tuple or len(q.enclosure)!=2 or any(type(x) is not F for x in q.enclosure):fail('quantity enclosure type','MALFORMED_PROOF')
        enclosure=tuple(F(x) for x in q.enclosure)
    return QuantityResult(str(q.name),q.numerator,q.norm_squared,None if q.binary64 is None else tuple(q.binary64),None if q.decimal is None else tuple(q.decimal),q.accuracy,q.accuracy_met,enclosure)
def copy_result(result,query):
    from ._result import CertifiedHydrostaticsResult,Contacts,SectionResult,Topology,TopologyLoop
    if type(result) is not CertifiedHydrostaticsResult:fail('unexpected result type','MALFORMED_PROOF')
    if type(result.moments) is not tuple or len(result.moments)!=3 or type(result.sections) is not tuple or len(result.sections)!=1 or result.sections[0].side!='plane':fail('curved response quantity/section cardinality','MALFORMED_PROOF')
    sections=[]
    for s in tuple(result.sections):
        if type(s) is not SectionResult or (s.quantities is not None and (type(s.quantities) is not tuple or len(s.quantities)!=6)):fail('section field type/cardinality','MALFORMED_PROOF')
        t=s.topology
        topology=None if t is None else Topology(tuple(TopologyLoop(tuple(l.identity),l.role) for l in t.loops),tuple((tuple(a),tuple(tuple(h) for h in holes)) for a,holes in t.components),tuple(tuple(x) for x in t.cut_points))
        sections.append(SectionResult(s.side,s.occupancy,s.section,None if s.quantities is None else tuple(copy_quantity(x) for x in s.quantities),topology))
    return CertifiedHydrostaticsResult(query,result.occupancy,result.generic,Contacts(tuple(result.contacts.vertices),tuple(result.contacts.edges),tuple(result.contacts.faces)),copy_quantity(result.volume),tuple(copy_quantity(q) for q in result.moments),tuple(sections),{},bytes(result.proof),None)

def metadata(value,meter,path='execution'):
    # Counts/core are untrusted and have no influence on mathematical acceptance.
    plain(value,meter)
    if type(value) is not dict:fail('execution metadata must be an object','MALFORMED_PROOF')
    for key,v in value.items():
        if key=='core':
            if type(v) is not dict:fail('invalid execution core','MALFORMED_PROOF')
            for name,ends in v.items():
                if type(ends) is not list or len(ends)!=2:fail('invalid core enclosure','MALFORMED_PROOF')
                lo,hi=map(rational,ends)
                if lo>hi:fail('invalid core enclosure order','MALFORMED_PROOF')
        elif type(v) is not int or v<0:fail('invalid execution counter','MALFORMED_PROOF')

def bounded_decimal(text,ar):
    # Inspect size and grammar before Decimal/Fraction or exponent expansion.
    policy=ar.policy;limit=policy['max_integer_digits']
    if type(text) is not str or len(text)>limit+16:fail('decimal digit limit','MALFORMED_PROOF')
    match=re.fullmatch(r'(-?)([0-9]+)(?:\.([0-9]+))?(?:[eE]([+-]?[0-9]+))?',text)
    if match is None:fail('invalid decimal grammar','MALFORMED_PROOF')
    negative,whole,fraction,exponent=match.groups();fraction=fraction or '';exponent=exponent or '0'
    if len(exponent.lstrip('+-'))>9:fail('decimal exponent limit','MALFORMED_PROOF')
    exponent=int(exponent)-len(fraction);digits=whole+fraction
    if len(digits)>limit or abs(exponent)>limit or (len(digits)+abs(exponent))*4>policy['max_bits']:fail('decimal precision limit','BIT_LENGTH_EXCEEDED')
    value=0
    for offset in range(0,len(digits),9):
        chunk=digits[offset:offset+9];ar.tick('verify_decimal_chunk');value=value*10**len(chunk)+int(chunk);ar.fit((F(value),))
    ar.tick('verify_decimal_scale');scale=10**abs(exponent);ar.fit((F(scale),))
    result=F(value*scale) if exponent>=0 else F(value,scale)
    return -result if negative else result


def check_quantity(name,claim,item,value,query,ar,default,exact_value=None):
    if item.name!=name:fail('public quantity name/order differs','MALFORMED_PROOF')
    if type(claim) is not dict:fail('quantity object required','MALFORMED_PROOF')
    allowed={'enclosure','exact','binary64','decimal','accuracy','accuracy_met'}
    if set(claim)-allowed:fail('unknown quantity field','MALFORMED_PROOF')
    if 'enclosure' not in claim:fail('curved quantity lacks enclosure','MALFORMED_PROOF')
    enc=claim['enclosure']
    if type(enc) is not dict or set(enc)!={'lo','hi'}:fail('invalid enclosure','MALFORMED_PROOF')
    lo,hi=rational(enc['lo']),rational(enc['hi'])
    if lo>hi:fail('reversed enclosure','MALFORMED_PROOF')
    accuracy=query.accuracy(name);target=accuracy if accuracy is not None else default
    mismatch(claim.get('accuracy')==(None if accuracy is None else pair(accuracy)) and claim.get('accuracy_met') is (True if accuracy is not None else None),'accuracy flag','incorrect accuracy flag')
    mismatch(hi-lo<=target,'accuracy flag','claim exceeds accuracy')
    mismatch(item.enclosure==(lo,hi),'claim','result enclosure differs from proof')
    if value.width:
        if 'exact' in claim and exact_value is None:fail('exact field on enclosed quantity','MALFORMED_PROOF')
        mismatch(hi>lo,'enclosure','zero width false exact enclosure')
        if value.width>(hi-lo)/3:fail('verifier enclosure insufficiently refined','BUDGET_EXHAUSTED')
    if 'exact' in claim:
        exact=claim['exact']
        if type(exact) is not dict or set(exact)!={'numerator','norm_squared'}:fail('invalid exact form','MALFORMED_PROOF')
        numerator=rational(exact['numerator']);norm=None if exact['norm_squared'] is None else rational(exact['norm_squared'])
        mismatch((value.lo==value.hi and norm in (None,F(1)) and numerator==value.lo) or exact_value==(numerator,norm),'exact','incorrect exact quantity')
    if not (lo<=value.lo and value.hi<=hi) and not (value.hi<lo or value.lo>hi):fail('independent enclosure overlaps claim without inclusion proof','BUDGET_EXHAUSTED')
    mismatch(lo<=value.lo and value.hi<=hi,'enclosure','independent enclosure outside claim')
    forms=set(query.forms)
    for form in ('binary64','decimal'):
        if (form in claim)!=(form in forms):fail('quantity forms differ from request','MALFORMED_PROOF')
        if form not in claim:continue
        v=claim[form]
        try:
            if form=='binary64':
                a,b=float.fromhex(v['lo']),float.fromhex(v['hi'])
                if not math.isfinite(a) or not math.isfinite(b):raise ValueError
                al,bh=F(a),F(b)
                mismatch(a==float.fromhex(item.binary64[0].hex()) and b==item.binary64[1],'claim','result binary64 differs')
            else:
                al,bh=bounded_decimal(v['lo'],ar),bounded_decimal(v['hi'],ar);advertised=rational(v['width'])
                mismatch(advertised==target and bh-al<=advertised,'enclosure','decimal advertised width differs from actual/requested bound')
            mismatch(al<=lo<=hi<=bh and bh-al<=target,'enclosure',form+' fails outward bound or accuracy')
        except (ValueError,TypeError,KeyError,AttributeError,OverflowError):fail('malformed '+form,'MALFORMED_PROOF')

def require_positive_full(full):
    if full.lo<=0<full.hi:fail('full-volume sign remains unresolved','BUDGET_EXHAUSTED')
    mismatch(full.lo>0,'model','solid has nonpositive full volume')


def verify_curved(model,faces,query,result,expected,*,budget=None,cancellation_check=None):
    """Verify a convention-2 claim using an independent geometry reconstruction."""
    from ._query import HalfSpaceQuery
    from ._prepare import CertifiedHydrostaticsPolicy
    from ._verify import VerifiedHydrostatics
    from ..serialization import _qualified_model_state
    meter=Meter(budget if budget is not None else WorkBudget(None,65536),cancellation_check)
    copied=False;arithmetic_token=_active.set(None)
    try:
        meter.unit('verify_start')
        if not isinstance(expected,Mapping):raise TypeError('expected must be a mapping')
        if type(expected.get('units')) is not str:raise TypeError('expected units must be a string')
        policy=expected.get('policy')
        policy=policy.as_dict() if isinstance(policy,CertifiedHydrostaticsPolicy) else dict(policy)
        policy=plain(policy,meter)
        mismatch(expected.get('convention_version')=='2' and policy.get('convention_version')=='2',
                 'convention','expected convention and preparation policy differ')
        ids=tuple(sorted(faces))
        if not ids or any(type(x) is not int for x in ids) or len(set(ids))!=len(ids):raise ValueError('invalid face set')
        if type(query) is not HalfSpaceQuery:raise TypeError('HalfSpaceQuery required')
        q=HalfSpaceQuery(tuple(query.normal),query.offset,tuple(query.waterplane_origin),tuple(tuple(x) for x in query.waterplane_axes),tuple(query.absolute_accuracy),tuple(query.forms))
        rebuilt=copy_result(result,q);raw=bytes(result.proof)
        copied=True
        mismatch(type(result.query) is HalfSpaceQuery and structural(result.query.as_dict(),q.as_dict()),'query','public result query differs')
        if len(raw)>policy['max_proof_bytes']:fail('proof byte limit exceeded','MALFORMED_PROOF')
        meter.unit('verify_proof_bytes',max(1,(len(raw)+65535)//65536))
        def integer(text):
            negative=text.startswith('-');digits=text[1:] if negative else text
            if len(digits)>policy['max_integer_digits']:fail('integer digit limit exceeded','MALFORMED_PROOF')
            value=0
            for offset in range(0,len(digits),9):
                piece=digits[offset:offset+9];meter.unit('verify_integer_chunk');value=value*10**len(piece)+int(piece);meter.fit(value)
                if value.bit_length()>policy['max_bits']:fail('proof integer exceeds bit cap','BIT_LENGTH_EXCEEDED')
            return -value if negative else value
        try:doc=json.loads(raw,object_pairs_hook=no_duplicates,parse_int=integer,parse_constant=lambda _:fail('nonfinite JSON','MALFORMED_PROOF'))
        except (ValueError,UnicodeError,RecursionError):fail('invalid JSON proof','MALFORMED_PROOF')
        doc=plain(doc,meter)
        if set(doc)!={'artifact','model','convention','preparation_policy','query','result'}:fail('unexpected proof fields','MALFORMED_PROOF')
        if type(doc['model']) is not dict or set(doc['model'])!={'model_id','revision','document_checksum','face_ids','snapshot_digest'}:fail('unexpected model identity fields','MALFORMED_PROOF')
        mismatch(structural(doc['query'],q.as_dict()),'query','query differs from proof')
        mismatch(structural(doc['preparation_policy'],policy),'convention','preparation policy differs')
        mismatch(doc['convention'].get('convention_version')=='2' and doc['convention'].get('units')==expected.get('units'),'convention','convention or units mismatch')
        convention={'area_element':'true area element dA of the plane; waterplane quantities are numerator/|n|','body_frame':'model coordinates are the body frame; no rotation, no normalisation','convention_version':'2','half_space':'n.r <= d with exact rational n, d (unnormalised)','limits':'below: n.r = d - eps, above: n.r = d + eps, eps -> 0+; V and J use the below convention','loop_identity':'sorted ids of the crossed shell edges','moment_order':['A','Mu','Mv','Iuu','Ivv','Iuv'],'units':expected.get('units'),'curved':{'binary64':'outward rounding of the claim','claim':'enclosure [lo, hi] = core +- width of the core; width of the claim <= the accuracy','cut_point_identity':'(edge id, ordinal along the edge by increasing knot parameter u)','decimal':'outward rounding of the claim','events':'generic planes only: a plane through a vertex or edge, tangent, or containing a ruling is refused CURVED_SECTION_DEGENERACY','exact':'present only when the computation was exact (no algebraic or enclosed term)','orientation':'model loops anticlockwise about the outward normal; V_f = -(1/3) oint F_V du; kept cut-graph portion directed -sign(beta) sigma_chart; cap = cut pieces reversed'}}
        mismatch(structural(doc['convention'],convention),'convention','hydrostatics convention differs')
        artifact=doc['artifact'];files=artifact.get('files');wanted=expected.get('artifact_files')
        if wanted is None:
            a=expected.get('artifact');a=a.as_dict() if hasattr(a,'as_dict') else a;wanted=a['files']
        mismatch(structural(files,[list(x) for x in wanted]),'artifact','artifact hashes differ from expected')
        root=Path(__file__).resolve().parents[1];actual=[]
        for path in sorted(root.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':
                meter.unit('verify_artifact_file');hasher=hashlib.sha256()
                with path.open('rb') as stream:
                    while chunk:=stream.read(65536):meter.unit('verify_artifact_bytes');hasher.update(chunk)
                actual.append([path.relative_to(root).as_posix(),hasher.hexdigest()])
        mismatch(actual==files,'artifact','runtime artifact differs')
        from .. import __version__
        try:
            version=importlib.metadata.version('anygeometry');requirements=importlib.metadata.requires('anygeometry') or []
        except importlib.metadata.PackageNotFoundError:
            version=None
            requirements=tomllib.loads((root.parents[1]/'pyproject.toml').read_text(encoding='utf-8'))['project'].get('dependencies',[])
        dependencies=[]
        for requirement in requirements:
            if 'extra' in requirement:continue
            name=re.split(r'[\s<>=!~;\[]',requirement,maxsplit=1)[0].lower().replace('_','-')
            try:depversion=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:depversion=None
            dependencies.append([name,depversion])
        runtime={'distribution':'anygeometry','distribution_version':version,'package_version':__version__,'files':actual,'dependencies':sorted(dependencies),'python':platform.python_implementation()+' '+platform.python_version()}
        mismatch(structural(artifact,runtime),'artifact','runtime artifact metadata differs')
        meter.unit('verify_model_entities',len(model.vertices)+len(model.edges)+len(model.faces))
        if getattr(model,'_transaction_journal',None) is not None:fail('live model transaction is in progress','CONCURRENT_MUTATION')
        before=(str(model.model_id),int(model.revision));document=_qualified_model_state(model);check=dict(document);check.pop('checksum',None);checksum=digest(check)
        mismatch(document.get('checksum',{}).get('value')==checksum,'model','public document checksum mismatch')
        ident=doc['model'];mismatch(ident.get('model_id')==before[0] and ident.get('revision')==before[1] and ident.get('document_checksum')==checksum and structural(ident.get('face_ids'),list(ids)),'model','model identity differs')
        ar=Arithmetic(meter,policy);rec=Reconstruction(model,ids,ar);plain(rec.snapshot,meter);snapshot_digest=digest(rec.snapshot)
        mismatch(ident.get('snapshot_digest')==snapshot_digest,'snapshot','snapshot differs from live model')
        from ._ch3_verify_embedding import Embedding
        claimed=doc['result'];claim=rebuilt.claim(with_work=False)
        if set(claimed)!={'contacts','execution','generic','moments','occupancy','sections','volume','work'}:fail('unexpected result fields','MALFORMED_PROOF')
        if set(claimed['moments'])!={'Jx','Jy','Jz'}:fail('incomplete moment claim','MALFORMED_PROOF')
        work=claimed['work']
        if type(work) is not dict or set(work)!={'operations','units'} or type(work['operations']) is not int or work['operations']<0 or type(work['units']) is not dict or any(type(v) is not int or v<0 for v in work['units'].values()):fail('malformed work counts','MALFORMED_PROOF')
        early=[('V',claimed['volume'])]+list(claimed['moments'].items())
        for section in claimed['sections'].values():
            if section['quantities'] is not None:early.extend(section['quantities'].items())
        default=F(1,10**policy['default_enclosure_digits'])
        for section in claimed['sections'].values():
            if (section['quantities'] is not None)!=(section['section']=='PRESENT'):fail('section quantities presence mismatch','MALFORMED_PROOF')
        for name,c in early:
            enclosure=c.get('enclosure')
            if type(enclosure) is not dict or set(enclosure)!={'lo','hi'}:fail('missing or malformed enclosure','MALFORMED_PROOF')
            lo,hi=rational(enclosure['lo']),rational(enclosure['hi'])
            if 'decimal' in c:
                bounded_decimal(c['decimal']['lo'],ar);bounded_decimal(c['decimal']['hi'],ar)
            if lo>hi:fail('reversed enclosure','MALFORMED_PROOF')
            accuracy=q.accuracy(name);target=default if accuracy is None else accuracy
            mismatch(c.get('accuracy')==(None if accuracy is None else pair(accuracy)) and c.get('accuracy_met') is (None if accuracy is None else True) and hi-lo<=target,'accuracy flag','accuracy declaration differs from requested bound')
        for c in (claim,):c.pop('execution',None);c.pop('work',None)
        supplied={k:v for k,v in claimed.items() if k not in ('work','execution')}
        mismatch(structural(claim,supplied),'claim','public response differs from proof')
        if 'execution' in claimed:metadata(claimed['execution'],meter)
        # Width derives only from the public claim, never from execution.core.
        quantities=[('V',claimed['volume'],rebuilt.volume)]+[(n,claimed['moments'][n],x) for n,x in zip(('Jx','Jy','Jz'),rebuilt.moments)]
        for section in rebuilt.sections:
            if section.quantities is not None:
                for name,item in zip(('A','Mu','Mv','Iuu','Ivv','Iuv'),section.quantities):quantities.append((name,claimed['sections'][section.side]['quantities'][name],item))
        widths=[]
        for name,c,item in quantities:
            if 'enclosure' not in c:fail('missing quantity enclosure','MALFORMED_PROOF')
            lo,hi=rational(c['enclosure']['lo']),rational(c['enclosure']['hi'])
            if hi<lo:fail('reversed quantity enclosure','MALFORMED_PROOF')
            if hi>lo:widths.append(hi-lo)
        targets={name:(rational(c['enclosure']['hi'])-rational(c['enclosure']['lo']))/12 for name,c,_ in quantities}
        # No water quantities are requested for absent sections; retain a finite
        # internal allowance for their identically zero accumulators.
        for name in ('A','Mu','Mv','Iuu','Ivv','Iuv'):targets.setdefault(name,default/12)
        Embedding(rec).run()
        for refinement in range(policy['max_refinement_rounds']):
            allowances={name:(value if value else default/12)/2**refinement for name,value in targets.items()}
            ar.point_bits=max(getattr(ar,'point_bits',140),140+64*refinement)
            computation=Computation(rec,q,allowances)
            try:values,occupancy,topology=computation.run()
            except CertifiedHydrostaticsRefusal as exc:
                if exc.category==RefusalCategory.BIT_LENGTH_EXCEEDED:fail('verification output precision exceeds arithmetic budget','BUDGET_EXHAUSTED')
                raise
            if all(values[name].width<=(rational(c['enclosure']['hi'])-rational(c['enclosure']['lo']))/3 and ((rational(c['enclosure']['lo'])<=values[name].lo and values[name].hi<=rational(c['enclosure']['hi'])) or values[name].hi<rational(c['enclosure']['lo']) or values[name].lo>rational(c['enclosure']['hi'])) for name,c,_ in quantities):break
        else:fail('output enclosure refinement exhausted','BUDGET_EXHAUSTED')
        mismatch(claimed['occupancy']==occupancy and claimed['generic'] is True,'occupancy','incorrect occupancy/generic flag')
        mismatch(structural(claimed['contacts'],{'vertices':[],'edges':[],'faces':[]}),'contacts','incorrect contacts')
        if set(claimed['sections'])!={'plane'}:fail('generic curved section must use plane','MALFORMED_PROOF')
        section=claimed['sections']['plane'];mismatch(section['occupancy']==occupancy and section['section']==('PRESENT' if topology else 'ABSENT'),'section','section presence mismatch')
        mismatch(structural(section['topology'],topology),'topology','section topology differs')
        default=F(1,10**policy['default_enclosure_digits'])
        for name,c,item in quantities:check_quantity(name,c,item,values[name],q,ar,default,computation.exact_values.get(name))
        # Positive NURBS weights put every physical point in the convex hull
        # of its controls. This independently proves the auxiliary plane dry.
        full=values['V']
        if occupancy!='FULL':
            z=[p[2] for p in rec.positions.values()]
            for fid in rec.supports:z.extend(F(p[2]) for row in rec.faces[fid].surface.controls for p in row)
            fq=HalfSpaceQuery((0,0,1),max(z)+1,forms=('decimal',))
            fc=Computation(rec,fq,default/4);fc.wanted={'V'}
            try:full_values,full_occupancy,_=fc.run()
            except CertifiedHydrostaticsRefusal as exc:
                if exc.category==RefusalCategory.BIT_LENGTH_EXCEEDED:fail('full-volume precision exceeds verification budget','BUDGET_EXHAUSTED')
                raise
            mismatch(full_occupancy=='FULL','model','full-volume support bound failed');full=full_values['V']
        require_positive_full(full)
        try:after=_qualified_model_state(model,expected_checksum=checksum,changed_message='verification source binding changed')
        except ValueError as exc:raise CertifiedHydrostaticsRefusal(RefusalCategory.CONCURRENT_MUTATION,'model changed during verification',phase='verify') from exc
        after_check=dict(after);after_check.pop('checksum',None)
        if getattr(model,'_transaction_journal',None) is not None or before!=(str(model.model_id),int(model.revision)) or digest(after_check)!=checksum:fail('model changed during verification','CONCURRENT_MUTATION')
        rebuilt=replace(rebuilt,work=plain(claimed['work'],meter),execution=plain(claimed.get('execution'),meter))
        present=section['section']=='PRESENT'
        returned={k:v for k,v in values.items() if present or k in ('V','Jx','Jy','Jz')}
        exact={k:v.lo for k,v in returned.items() if v.width==0};exact.update(occupancy=occupancy,generic=True,contacts={'vertices':(),'edges':(),'faces':()},sides={'plane':{'occupancy':occupancy,'section':section['section'],'quantities':({k:v for k,v in computation.exact_values.items() if k in ('A','Mu','Mv','Iuu','Ivv','Iuv')} if present else None),'topology':topology}})
        if all(values[k].width==0 for k in ('Jx','Jy','Jz')):exact['J']=tuple(values[k].lo for k in ('Jx','Jy','Jz'))
        enclosures={k:(v.lo,v.hi) for k,v in returned.items()}
        full_volume=full.lo if full.width==0 else (full.lo,full.hi)
        return VerifiedHydrostatics(rebuilt,q,ids,before[0],before[1],checksum,snapshot_digest,digest(files),hashlib.sha256(raw).hexdigest(),full_volume,exact,meter.budget.counts(),enclosures=enclosures)
    except CertifiedHydrostaticsRefusal as exc:
        raise exc.complete(phase='verify',work=meter.budget.counts())
    except RecursionError as exc:
        raise CertifiedHydrostaticsRefusal(RefusalCategory.BUDGET_EXHAUSTED,'verifier recursion budget exhausted',phase='verify',work=meter.budget.counts()) from exc
    except (KeyError,IndexError,AttributeError,ZeroDivisionError,OverflowError) as exc:
        raise CertifiedHydrostaticsRefusal(RefusalCategory.MALFORMED_PROOF,'malformed curved proof or public response',phase='verify',work=meter.budget.counts()) from exc
    except TypeError as exc:
        if not copied:raise
        raise CertifiedHydrostaticsRefusal(RefusalCategory.MALFORMED_PROOF,'malformed curved proof value type',phase='verify',work=meter.budget.counts()) from exc
    except ValueError as exc:
        if not copied:raise
        raise CertifiedHydrostaticsRefusal(RefusalCategory.MALFORMED_PROOF,'invalid curved proof or model value',phase='verify',work=meter.budget.counts()) from exc

    finally:
        _active.reset(arithmetic_token)
