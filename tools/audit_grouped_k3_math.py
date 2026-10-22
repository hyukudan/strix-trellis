"""Local CPU-only audit of archived linked bytes; emit a fail-closed handoff."""
from pathlib import Path
import hashlib
import io
import json
import struct
import sys
import tarfile
import msgpack

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'reports/grouped-k3-math-data'
sys.path.insert(0,str(ROOT/'tools'))
from audit_grouped_k3_pilot import sections,manifest


def sha(b):return hashlib.sha256(b).hexdigest()


def metadata(elf):
    note=sections(elf)['.note'];p=0;found=[]
    while p<len(note):
        ns,ds,kind=struct.unpack_from('<III',note,p);p+=12
        name=note[p:p+ns];p+=(ns+3)&~3
        desc=note[p:p+ds];p+=(ds+3)&~3
        if name==b'AMDGPU\0' and kind==32:found.append(msgpack.unpackb(desc,raw=False))
    assert len(found)==1
    return found[0]['amdhsa.kernels']


def images(fat):
    marker=b'__CLANG_OFFLOAD_BUNDLE__';pos=0;result=[]
    while True:
        p=fat.find(marker,pos)
        if p<0:break
        q=p+len(marker);count,=struct.unpack_from('<Q',fat,q);q+=8;end=q
        for _ in range(count):
            offset,size,length=struct.unpack_from('<QQQ',fat,q);q+=24
            target=fat[q:q+length].decode();q+=length;end=max(end,p+offset+size)
            if 'gfx1151' in target:result.append(fat[p+offset:p+offset+size])
        pos=end
    return result


def main():
    handoff=dict(schema=1,scope='PRIVATE_SOURCE_CPU_BUILD_ONLY',root_review_required=True,
        gpu_executed=False,hip_executable_executed=False,turn_requests=0,
        baseline_archive_sha256='0b1453a3c9c25e3f7e4f12251d310a3a9f2c3b53c76d78c314e9624ba763e3b3',
        baseline_literal_lf_sha256='a5fbb60a5cde97442455e041e7f10c76397209cf2c335fc16600346c3da59503',
        baseline_dispatch='src/gemv_lds.hip::gemv_baseline_kernel<3,true,1,false>',
        public_abi=1,private_weight_fields=9,native_batch_bytes=2320,
        native_admission='OPEN',nodes={},
        gates={'P1-01':dict(status='OPEN',source_correspondence='CLOSED_CPU',effective_headers='CLOSED_BUILD',
                linked_math_instruction_identity='OPEN',native_exact_A_P_Y='OPEN',native_FP64_envelope='OPEN'),
            'P1-02':dict(status='OPEN',static_linked_ELF='CLOSED_BUILD',scalar_descriptor_selection='CLOSED_STATIC_FINITE_REVIEW',
                function_attributes_and_admission='OPEN',runtime_host_and_gpu_ledger='OPEN'),
            'P1-03':dict(status='OPEN',actual_owner_plan_graph='SOURCE_BUILD_ONLY',guard_and_first_error_mock='CLOSED_CPU_ONLY',
                mathematical_capture_graph_only_leases_twostream_LAST='OPEN',actual_allocation_recycle='NOT_IMPLEMENTED_NO_RECYCLE',
                native_enqueue_async_failure_observation='OPEN',native_allocation_failure_injection='NOT_CLAIMED')},
        native_qa=dict(status='SOURCE_BUILD_ONLY_NOT_ADMITTED',configurations=14,groups=[1,2,4,8,16,31,32],
            geometries=[[5120,2304,16],[2304,5120,4]],physical_budget_seconds=180,inner_seconds=145,
            budget_fit='UNMEASURED_ROOT_SPLIT_DECISION_IF_NEEDED',
            canonical_payload='cd ~/gpu-site/scratch-trellis/grouped-k3-math && ROOT_ADMISSION_SHA256=<reviewed_per_node_frozen_hash> bash experiments/grouped_k3_math/native_payload.sh --inside-root-lease',
            payload_requires_root_obtained_gpu_turn_any_lease=True,queue_command_executed=False),
        performance=dict(status='OPEN_NO_RUNNER_NO_MEASUREMENTS',selection_groups=[4,8,16,32],physical_campaigns_required=2,
            sweeps=11,minimum_disjoint_traversed_packed_bytes=256*1024**2,
            baseline='prepared original V13 3G stage sequence, same S, independent workspaces',
            comparison='both graphs or both direct, event and wall, min/median/p90/all losses, every clone before/after/LAST',
            caveat='QA residency across both geometries does not establish per-projection cold traversal'),
        resource_bounds=json.loads((DATA/'resource-bounds.json').read_text()),
        cpu_manifest_sha256=sha((DATA/'cpu-manifest.json').read_bytes()))
    source=manifest((DATA/'source.sha256').read_text())
    with tarfile.open(DATA/'source.tar') as tar:
        assert set(tar.getnames())==set(source)
        for name,pin in source.items():
            assert sha(tar.extractfile(name).read())==pin==sha((ROOT/name).read_bytes()),name
    for node in ('node1','node2'):
        archive=DATA/(node+'-evidence.tar.gz')
        with tarfile.open(archive) as tar:
            members={m.name:tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
        frozen=manifest(members['frozen.sha256'].decode())
        assert members['frozen.sha256']==(DATA/(node+'-frozen.sha256')).read_bytes()
        checked=0
        for name,raw in members.items():
            if name in frozen:assert sha(raw)==frozen[name],(node,name);checked+=1
        assert members['source.tar']==(DATA/'source.tar').read_bytes()
        assert members['source.sha256']==(DATA/'source.sha256').read_bytes()
        assert sha(members['lib/libstrix_trellis.a'])==handoff['baseline_archive_sha256']
        fat=sections(members['bin/math_native_qa'])['.hip_fatbin']
        assert fat==members['build/linked.fatbin']
        emitted=images(fat);assert len(emitted)==8
        counts=[]
        for k,elf in enumerate(emitted):
            assert elf==members[f'build/linked-{k:02d}.elf']
            counts.append(len(metadata(elf)))
        assert counts==[3,2,72,252,1,32,2,2]
        resources=json.loads(members['build/resources.json'])
        assert not resources['violations']
        kernels=metadata(emitted[0]);assert len(kernels)==3
        for k,r in zip(kernels,resources['resources']):
            assert k['.name']==r['name']
            assert k['.args']==[{'.offset':0,'.size':2320,'.value_kind':'by_value'}]
            for key in ('kernarg_segment_size','kernarg_segment_align','group_segment_fixed_size','private_segment_fixed_size',
                        'wavefront_size','max_flat_workgroup_size','sgpr_count','vgpr_count','sgpr_spill_count','vgpr_spill_count'):
                assert k['.'+key]==r[key],key
        scalar=json.loads(members['build/scalar-selection.json']);assert scalar['mutation_tests_rejected']==6
        assert scalar['effective_preprocessed_sha256']==sha(members['build/effective-source.ii'])
        handoff['nodes'][node]=dict(remote_scope='~/gpu-site/scratch-trellis/grouped-k3-math',
            frozen_manifest_sha256=sha(members['frozen.sha256']),linked_binary_sha256=sha(members['bin/math_native_qa']),
            library_sha256=sha(members['lib/libstrix_trellis.a']),math_ELF_sha256=sha(emitted[0]),
            evidence_archive_sha256=sha(archive.read_bytes()),locally_verified_archived_frozen_files=checked,
            total_linked_images=8,total_linked_kernels=sum(counts),math_kernels=3,original_archive_kernels=sum(counts)-3,
            static_math_resources=resources['resources'],scalar_selection=scalar,
            cpu_data_verified_remotely=True,full_frozen_manifest_verified_remotely=True)
        print('PASS',node,'actual linked binary/fatbin/all8ELFs/MessagePack/headers/static resources/source provenance',checked,'archived frozen members')
    (DATA/'handoff.json').write_text(json.dumps(handoff,indent=2)+'\n',newline='\n')
    print('PASS CPU archive audit; all three mathematical P1 remain OPEN; no GPU admission')


if __name__=='__main__':main()
