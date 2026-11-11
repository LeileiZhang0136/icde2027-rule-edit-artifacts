#!/usr/bin/env python3
"""Build a fresh, executable-rule main benchmark draft.

This builder does not read legacy case or gold files.  The three carried-over
events are re-formalized from their upper-rule mechanisms only; every domain,
case family, fact pattern, and surface form in the output is newly generated.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import random
import re
from collections import Counter, defaultdict
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

VERSION = "main-experiment-formal-v1.1"
SEED = 20261012
EFFECTIVE_OLD = "2026-01-01"
EFFECTIVE_NEW = "2027-01-01"
BASE_SPLITS = {"train": 3, "development": 1}
TEST_DOMAINS_PER_EVENT = {
    "RTE_AGE_18_TO_20": 10,
    "RTE_BIOMETRIC_VOICE_EXPANSION": 9,
    "RTE_MEANINGFUL_HUMAN_REVIEW": 10,
    "RTE_VALIDITY_180_TO_90": 10,
    "RTE_EVIDENCE_1_TO_2": 9,
    "RTE_DEFAULT_TO_OPT_IN": 9,
    "RTE_EMERGENCY_EXCEPTION_NARROWING": 9,
    "RTE_PRIORITY_UPPER_OVER_LOCAL": 9,
    "RTE_TRANSITION_PENDING_CASES": 10,
}
EXPECTED_TEST_FAMILIES = 1700
VARIANTS = ("easy", "medium", "hard")
N_FAMILIES_PER_DOMAIN = 20
HUMAN_REVIEW_SCOPE = (
    "本任务中的‘是否需要人工复核’专指上层规则定义的、由独立审查人实施的实质性审查：审查人查看个案材料和系统理由、"
    "有权修改或推翻系统建议，并记录审查理由。值班员手动操作或指令核对、监护人共同确认、技术检查/复检、"
    "常规资格核验、补充证据及申诉步骤不自动计为此标签；如适用，仍须作为当前处置或程序要求说明。"
)

# Existing events are preserved at the mechanism/text level.  No existing
# downstream cases, labels, or domain examples are imported.
EVENTS: list[dict[str, Any]] = [
    {
        "event_id": "RTE_AGE_18_TO_20", "kind": "age_threshold",
        "effective_date": EFFECTIVE_OLD,
        "title": "独立实施相关行为的年龄标准由18岁调整为20岁",
        "old_rule": "受控辖区J中，独立实施规则所覆盖行为的年龄标准为18岁。年龄按相关行为发生日计算。",
        "new_rule": "自2026-01-01起，受控辖区J中，独立实施规则所覆盖行为的年龄标准调整为20岁。年龄按相关行为发生日计算。",
        "mechanism": {"old_threshold_years": 18, "new_threshold_years": 20, "age_basis": "completed_years_on_action_date"},
    },
    {
        "event_id": "RTE_BIOMETRIC_VOICE_EXPANSION", "kind": "scope_expansion",
        "effective_date": EFFECTIVE_OLD,
        "title": "敏感生物识别信息定义扩展至声纹和语音身份向量",
        "old_rule": "受控辖区J的敏感生物识别信息包括用于识别个人的面部模板和指纹模板，不包括声纹或语音身份向量。",
        "new_rule": "自2026-01-01起，受控辖区J的敏感生物识别信息还包括用于识别个人的声纹及语音身份向量。仅转写语音内容、未推断身份且未生成身份特征的普通语音处理不属于本次扩展。",
        "mechanism": {"old_sensitive_features": ["face_template", "fingerprint_template"], "added_feature": "voiceprint", "requires_identity_inference": True, "requires_generated_identity_feature": True},
    },
    {
        "event_id": "RTE_MEANINGFUL_HUMAN_REVIEW", "kind": "review_strengthening",
        "effective_date": EFFECTIVE_OLD,
        "title": "高影响自动化决策的人审要求由形式点击改为实质审查",
        "old_rule": "受控辖区J中，高影响自动化决定可由工作人员形式点击确认后生效。",
        "new_rule": "自2026-01-01起，高影响自动化决定生效前，审查人必须查看个案材料和系统理由、具有修改或推翻建议的权限，并记录审查理由。仅点击确认不构成实质审查；完成审查前不得执行不利决定。仅提供低影响排序或文书提示且不决定资格的系统不在本规则范围内。",
        "mechanism": {"old_standard": "formal_confirmation", "new_standard": "meaningful_review", "meaningful_elements": ["materials_checked", "system_reason_checked", "reviewer_can_override", "reason_recorded"]},
    },
    {
        "event_id": "RTE_VALIDITY_180_TO_90", "kind": "validity_window_shortening",
        "effective_date": EFFECTIVE_NEW,
        "title": "资格核验有效期由180日缩短为90日",
        "old_rule": "在受控辖区J，完成一次资格核验后，该核验结果在180日内有效。",
        "new_rule": "自2027-01-01起，在受控辖区J，完成一次资格核验后，该核验结果仅在90日内有效；超过90日须重新核验后方可继续依赖该结果。",
        "mechanism": {"old_validity_days": 180, "new_validity_days": 90, "age_basis": "days_since_last_check"},
    },
    {
        "event_id": "RTE_EVIDENCE_1_TO_2", "kind": "evidence_threshold_increase",
        "effective_date": EFFECTIVE_NEW,
        "title": "关键事实的独立证据门槛由一项提高为两项",
        "old_rule": "受控辖区J中，关键事实经至少一项来源独立且可核验的证据支持，即达到最低证据门槛。",
        "new_rule": "自2027-01-01起，受控辖区J中，关键事实须经至少两项彼此独立且可核验的证据支持，才达到最低证据门槛；同一来源的重复记录不计为两项。",
        "mechanism": {"old_min_independent_sources": 1, "new_min_independent_sources": 2},
    },
    {
        "event_id": "RTE_DEFAULT_TO_OPT_IN", "kind": "permission_default_reversal",
        "effective_date": EFFECTIVE_NEW,
        "title": "默认许可改为明确授权",
        "old_rule": "受控辖区J中，除非当事人明确拒绝，否则可依照本领域规则开展所述非生物识别数据的二次用途处理。",
        "new_rule": "自2027-01-01起，受控辖区J中，只有取得当事人针对该项二次用途的明确肯定授权后，方可开展该处理；未回应不视为授权。",
        "mechanism": {"old_default_allowed_unless_opt_out": True, "new_requires_affirmative_opt_in": True},
    },
    {
        "event_id": "RTE_EMERGENCY_EXCEPTION_NARROWING", "kind": "exception_narrowing",
        "effective_date": EFFECTIVE_NEW,
        "title": "紧急例外由一般紧迫情形收窄为不可替代的即时安全风险",
        "old_rule": "受控辖区J中，受限制操作在出现紧迫情形时可适用紧急例外。",
        "new_rule": "自2027-01-01起，只有存在即时人身安全风险且没有安全替代方案时，紧急例外才允许受限制操作；仅有时间紧迫或服务不便不足以适用例外。",
        "mechanism": {"old_exception": "urgent", "new_exception": ["urgent", "immediate_safety_risk", "no_safe_alternative"]},
    },
    {
        "event_id": "RTE_PRIORITY_UPPER_OVER_LOCAL", "kind": "conflict_priority_change",
        "effective_date": EFFECTIVE_NEW,
        "title": "规则冲突时由领域默认优先改为上层安全限制优先",
        "old_rule": "受控辖区J中，领域操作规则与上层安全限制发生直接冲突时，按领域操作规则处理。",
        "new_rule": "自2027-01-01起，受控辖区J中，领域操作规则与上层安全限制发生直接冲突时，以上层安全限制为准；不存在直接冲突时仍按各自适用规则处理。",
        "mechanism": {"old_conflict_priority": "domain_rule", "new_conflict_priority": "upper_safety_rule"},
    },
    {
        "event_id": "RTE_TRANSITION_PENDING_CASES", "kind": "transition_scope_expansion",
        "effective_date": EFFECTIVE_NEW,
        "title": "新标准由仅适用于新申请扩展至尚未执行的待处理申请",
        "old_rule": "受控辖区J中，2027-01-01起的新标准仅适用于该日或之后提交的申请；此前提交的申请按旧标准处理。",
        "new_rule": "自2027-01-01起，除已执行完毕的事项外，所有尚未执行的待处理申请均须满足新标准，包括生效日前提交的申请；已执行事项不追溯撤销。",
        "mechanism": {"old_scope": "filed_on_or_after_effective_date", "new_scope": "all_unexecuted_pending_on_or_after_effective_date", "no_retroactive_reversal": True},
    },
]


def domain(domain_id: str, name: str, split: str, mode: str, chain: str,
           action: str, positive: str, negative: str, fallback: str,
           invariant: str, secondary: str = "", exception: bool = False,
           local_rule: str = "", impact_fact: str = "",
           impact_true_text: str = "", impact_false_text: str = "",
           secondary_gate_label: str = "", grandfathered_credential_name: str = "") -> dict[str, Any]:
    return {"domain_id": domain_id, "name": name, "split": split, "mode": mode,
            "decision_chain": chain, "action": action, "positive_decision": positive,
            "negative_decision": negative, "fallback_action": fallback,
            "invariant_obligation": invariant, "secondary_gate": secondary,
            "has_downstream_exception": exception, "local_rule": local_rule,
            "impact_fact": impact_fact, "impact_true_text": impact_true_text,
            "impact_false_text": impact_false_text, "secondary_gate_label": secondary_gate_label,
            "grandfathered_credential_name": grandfathered_credential_name}


# The profiles deliberately vary the affected decision-chain link, secondary
# gate, fallback route, and invariant.  Prior-case domains are used only as an exclusion list.
DOMAINS: dict[str, list[dict[str, Any]]] = {
    "RTE_AGE_18_TO_20": [
        domain("age_robotics_permit", "机器人维护区的独立操作许可", "train", "permit", "年龄资格→设备资质→独立/模拟操作许可", "独立操控设备", "independent_operation_authorized", "simulation_only", "改走模拟器；未达资质时不启动设备", "保留设备安全检查记录", "hardware_certificate"),
        domain("age_research_export", "研究参与者的数据导出授权", "train", "authority", "年龄资格→数据主体授权能力→本人/共同授权后导出", "导出本人的研究数据", "self_authorized_export", "coapproval_required", "等待研究监护协调员共同确认", "记录导出范围和授权来源", "coapproval_done"),
        domain("age_field_lead", "野外考察的领队任命", "train", "role", "年龄资格→领队角色分派→领队/协作员职责", "担任野外考察领队", "lead_role_assigned", "assistant_role_assigned", "安排为协作员而非领队", "保留行程风险评估记录", "field_training_complete", True, "新年龄标准生效前已签发且仍有效的领队证，在到期前继续有效；该过渡例外不免除野外训练条件。"),
        domain("age_microgrid_dispatch", "微电网控制台的调度权限", "development", "dispatch", "年龄资格→调度训练→直接调度/影子席位/暂停", "执行调度指令", "direct_dispatch_authorized", "dispatch_not_authorized", "转入影子席位或等待值班主管", "记录每次调度指令和操作者", "dispatch_exam_passed"),
        domain("age_archive_lending", "特藏档案的外借资格", "test", "lending", "借阅方式→年龄资格→馆内阅览/外借审批", "将特藏原件带离阅览室", "offsite_lending_allowed", "onsite_only", "改为馆内阅览，不办理外借", "保留档案交接与归还记录", "", True, "新年龄标准生效前已批准且仍有效的外借许可，在到期前继续有效；馆内阅览始终可用。"),
        domain("age_remote_prototype", "远程原型设备的控制权", "test", "remote", "年龄资格→操控认证→远程控制/现场监督", "远程控制原型设备", "remote_control_authorized", "supervised_local_only", "仅在现场监督下操作", "记录远程会话和安全中止事件", "remote_operator_exam_passed"),
    ],
    "RTE_BIOMETRIC_VOICE_EXPANSION": [
        domain("voice_grid_dispatch", "语音身份向量触发的电网调度", "train", "biometric_dispatch", "身份特征范围→授权和保护措施→指令执行/值班员手动接管", "依据声音身份执行调度指令", "voice_dispatch_authorized", "manual_dispatch_required", "转交值班员手动执行指令", "记录指令来源和调度日志", "separate_consent+encrypted+purpose_match+retention_valid"),
        domain("voice_lab_entry", "共享实验室的声纹门禁", "train", "biometric_entry", "声纹识别→区域准入→实体卡替代", "凭声纹进入共享实验区", "voice_entry_authorized", "badge_entry_only", "改用实体门卡，由现场工作人员按门禁流程放行", "保留区域进出日志", "separate_consent+encrypted+purpose_match+retention_valid"),
        domain("voice_archive_profile", "声音身份驱动的档案个性化检索", "train", "biometric_profile", "身份推断→个性化检索→个人化/通用目录", "按个人声音档案排序检索结果", "personalized_search_allowed", "generic_search_only", "停用个人化排序，保留通用目录检索", "记录检索目的和排序配置", "separate_consent+purpose_match+retention_valid"),
        domain("voice_vehicle_release", "维修车队的语音身份放车", "development", "biometric_release", "声纹认证→车辆交接→放车/一次性口令替代", "确认维修车辆的领用人", "voice_release_authorized", "one_time_code_required", "通过一次性口令和交接员核验放车", "保留车辆交接与认证记录", "separate_consent+encrypted+purpose_match+retention_valid"),
        domain("voice_kiosk_records", "服务终端的语音身份调取个人记录", "test", "biometric_records", "身份特征范围→记录访问→柜员核验/自助调取", "从服务终端调取个人记录", "self_service_record_access", "staff_verification_required", "转由工作人员核验后调取", "记录每次个人记录访问", "separate_consent+encrypted+purpose_match+retention_valid"),
        domain("voice_transcript_routing", "仅转写内容的语音服务分流", "test", "biometric_transcript", "身份推断与特征生成→内容转写/身份匹配分流", "把语音内容转写后分配服务队列", "transcription_routing_allowed", "identity_matching_disabled", "仅做内容转写，不建立说话人身份向量", "保留工单分派理由", "identity_inference+identity_feature_created"),
    ],
    "RTE_MEANINGFUL_HUMAN_REVIEW": [
        domain("review_vendor_suspension", "自动化供应商停用", "train", "review_vendor", "关键服务是否依赖唯一供应方→停用影响判定→形式确认/实质审查→停用或暂缓", "停用供应商接口", "adverse_action_ready", "adverse_action_held", "暂停停用，进入人工审查队列", "保留审查材料和停用理由", "sole_critical_provider", False, "仅当该供应商是关键接口唯一合格来源时，停用才会中断核心服务并构成高影响；存在可用替代来源时按一般供应商维护流程处理。", impact_fact="sole_critical_provider", impact_true_text="关键接口目前只有一家合格供应方", impact_false_text="关键接口另有可用的合格供应方"),
        domain("review_scholarship_withdrawal", "奖助资格自动撤销", "train", "review_aid", "资助是否构成主要生活来源→撤销影响判定→查看证据并可改写→撤销或待审", "撤销奖助资格", "withdrawal_ready", "withdrawal_held", "不发撤销通知，先完成实质审查", "保留证据版本与审查理由", "primary_support_dependency", False, "仅当该资助是受助人主要生活来源时，自动撤销才构成高影响；有稳定替代来源时不满足本领域高影响条件。", impact_fact="primary_support_dependency", impact_true_text="该资助是受助人的主要生活来源", impact_false_text="受助人另有稳定的主要生活来源"),
        domain("review_content_restriction", "高影响内容账户限制", "train", "review_content", "拟限制范围→核心发布权限影响判定→实质审查/单条内容提示", "限制账号核心发布权限", "restriction_ready", "restriction_held", "限制前转人工审查，保留申诉入口", "保留处理记录与申诉通道", "account_level_restriction", False, "只有限制整个账号的核心发布权限才视为高影响；单条内容的提示或排序调整不构成本领域高影响处理。", impact_fact="account_level_restriction", impact_true_text="拟限制整个账号的核心发布权限", impact_false_text="拟处理仅针对单条内容且不限制账号发布权限"),
        domain("review_equipment_access", "高风险设备操作权限暂停", "development", "review_equipment", "设备用途是否安全关键→停权影响判定→技术审查→暂停或限制", "暂停设备操作权限", "suspension_ready", "suspension_held", "等待有权限的审查人查看风险说明", "保留设备安全事件和复核记录", "safety_critical_capability", False, "只有当设备用于关键安全控制时，自动暂停其操作权限才构成高影响；仅用于非关键测试的设备不适用该影响条件。", impact_fact="safety_critical_capability", impact_true_text="设备用于关键安全控制", impact_false_text="设备仅用于非关键测试"),
        domain("review_workspace_badge", "共享工作区的自动门禁撤销", "test", "review_badge", "场地是否为岗位唯一工作地点→撤销影响判定→临时通行/撤销", "撤销共享工作区门禁", "badge_revocation_ready", "badge_revocation_held", "审查完成前维持可撤销的临时访问状态", "保留访问与权限变更日志", "essential_site_access", False, "仅当该场地是岗位执行职责所需的唯一工作地点时，撤销门禁才构成高影响；有可用替代场地时不满足该条件。", impact_fact="essential_site_access", impact_true_text="该场地是执行当前岗位所需的唯一工作地点", impact_false_text="另有可用场地且不影响当前岗位职责"),
        domain("review_research_enrollment", "研究项目参与资格中止", "test", "review_research", "是否处于连续干预阶段→中止影响判定→参与者风险与数据核查→中止或暂缓", "中止参与者的研究项目资格", "enrollment_stop_ready", "enrollment_stop_held", "先由审查人复核材料和系统理由", "记录参与者通知与研究数据处置", "active_intervention_continuity", False, "仅当参与者正在接受需要连续观察的干预时，中止资格才构成高影响；行政随访阶段不满足此条件。", impact_fact="active_intervention_continuity", impact_true_text="参与者正在接受需要连续观察的干预", impact_false_text="参与者当前仅在行政随访阶段"),
    ],
    "RTE_VALIDITY_180_TO_90": [
        domain("valid_sensor_license", "环境传感器部署许可", "train", "validity_hold", "核验时效→部署许可→继续采集/重新核验", "部署环境传感器", "deployment_authorized", "deployment_waiting_for_recheck", "暂缓部署并重新核验许可证", "保留部署地点与校验记录", "site_approval_current"),
        domain("valid_machine_safety", "精密加工设备安全放行", "train", "validity_maintenance", "检查时效→安全状态→放行/技师复检", "启动精密加工设备", "machine_start_allowed", "machine_start_blocked", "锁定启动并安排技师复检", "保留检修和复检记录", "maintenance_complete"),
        domain("valid_dataset_reuse", "科研数据集的二次用途授权", "train", "validity_reconsent", "授权核验时效→用途资格→沿用授权/重新核验用途", "将数据集用于新的分析批次", "reuse_authorized", "reuse_review_required", "暂停二次使用并重新核验用途范围", "记录用途版本和授权版本", "purpose_unchanged"),
        domain("valid_supply_batch", "实验耗材批次的接收与隔离", "development", "validity_quarantine", "批次检验时效→收货状态→接收/隔离", "将耗材批次投入实验", "batch_release_allowed", "batch_quarantined", "隔离该批次，等待重新检验", "保留批次来源和检验日志", "storage_conditions_valid"),
        domain("valid_remote_credential", "远程维护凭证续期", "test", "validity_downgrade", "凭证时效→远程权限→全权限/只读模式", "远程写入控制系统", "remote_write_allowed", "read_only_mode", "降为只读并重新验证维护凭证", "保留远程操作和授权日志", "maintenance_scope_matches"),
        domain("valid_field_permit", "野外采样路线准入", "test", "validity_route", "路线风险核验时效→路线开放→开放/改走已核验路线", "进入指定野外采样路线", "route_entry_allowed", "route_entry_deferred", "暂不进入并改走已核验路线或等待复检", "记录路线选择与风险告知", "weather_route_safe"),
    ],
    "RTE_EVIDENCE_1_TO_2": [
        domain("evidence_material_origin", "档案材料来源核实", "train", "evidence_provenance", "来源证据数→来源链完整性→认证/待补证", "认证档案材料来源", "provenance_established", "provenance_unresolved", "暂停认证并补充独立来源", "保留来源链与证据编号", "source_chain_intact", False, "来源链连续且每项记录能追溯到原始保管方，才可认证档案来源。"),
        domain("evidence_fault_diagnosis", "设备故障原因认定", "train", "evidence_diagnosis", "独立诊断来源→故障复现→维修/观察", "确认设备故障并选择维修路径", "fault_supported", "diagnosis_pending", "先观察并补采独立诊断证据", "记录诊断依据与更换部件", "fault_reproduced", False, "故障须在独立复测中复现；仅有历史告警不够启动维修。"),
        domain("evidence_grant_eligibility", "项目资助条件核实", "train", "evidence_eligibility", "独立证明来源→预算可用性→资格确认/待核", "确认项目是否达到资助条件", "eligibility_supported", "eligibility_unverified", "暂不确认资格，等待第二来源核验", "记录资格判断理由", "budget_available", False, "即使申请条件成立，拨付额度也必须仍在本轮预算余额内。"),
        domain("evidence_water_quality", "水样异常结果确认", "development", "evidence_confirmation", "独立检测证据→采样链完整性→发布预警/继续检测", "发布水样异常预警", "alert_supported", "alert_unconfirmed", "暂缓正式预警并安排独立复测", "保留采样链和复测记录", "sampling_chain_intact", False, "只有采样封签和交接记录完整的检测结果才可用于正式预警。"),
        domain("evidence_safety_clearance", "设备维护后的安全放行", "test", "evidence_clearance", "独立放行证据→维修后负载测试→恢复/隔离", "解除设备检修隔离", "clearance_supported", "equipment_stays_isolated", "维持隔离并请求独立放行检查", "记录解除隔离的责任人与依据", "post_repair_test_passed", False, "维修完成后还须通过独立负载测试，才可解除设备隔离。"),
        domain("evidence_claim_resolution", "资源申领争议核验", "test", "evidence_claim", "独立记录→申领人匹配→确认/补充说明", "确认申领事实并分配资源", "claim_supported", "claim_unresolved", "暂缓最终分配并补充独立记录", "记录资源分配理由", "claimant_identity_matches", False, "申领记录必须与实际申请人身份匹配，才可最终分配资源。"),
    ],
    "RTE_DEFAULT_TO_OPT_IN": [
        domain("optin_sensor_reuse", "环境传感器数据的研究复用", "train", "permission_reuse", "肯定授权状态→复用许可→复用/仅本次服务", "把传感数据用于研究复用", "research_reuse_allowed", "service_only", "仅保留原始服务用途，不进入研究集", "记录授权用途和数据范围", "affirmative_authorization"),
        domain("optin_archive_digitization", "档案影像的训练用途授权", "train", "permission_training", "素材来源→用途授权→加入训练集/隔离素材", "将档案影像用于模型训练", "training_use_allowed", "training_use_blocked", "将素材排除出训练集并保留目录记录", "记录素材授权版本", "affirmative_authorization"),
        domain("optin_research_recontact", "研究参与者的后续联系", "train", "permission_recontact", "后续联系授权→联系范围→发送联系/停止联系", "发送后续研究邀请", "recontact_allowed", "no_recontact", "不发送邀请，只保留必要的项目通知", "记录联系目的和时间", "affirmative_authorization"),
        domain("optin_workspace_telemetry", "共享工作区行为日志分析", "development", "permission_analytics", "分析授权→使用粒度→统计分析/仅运行安全日志", "将工作区日志用于行为分析", "behavioral_analysis_allowed", "safety_logs_only", "停用行为分析，保留必要安全日志", "记录分析目的与访问范围", "affirmative_authorization"),
        domain("optin_audio_indexing", "非生物识别音频内容的检索索引", "test", "permission_indexing", "内容索引授权→语义检索→建立索引/不入索引", "将语音内容加入可检索索引", "content_indexing_allowed", "content_not_indexed", "保留原始工单，不建立跨工单检索索引", "记录索引字段和删除状态", "affirmative_authorization"),
        domain("optin_equipment_telemetry", "实验设备运行数据的跨项目共享", "test", "permission_sharing", "共享授权→接收项目边界→共享/本项目留存", "向另一个项目共享设备运行数据", "cross_project_share_allowed", "local_project_only", "数据留在原项目，不向外共享", "记录共享对象和数据范围", "affirmative_authorization"),
    ],
    "RTE_EMERGENCY_EXCEPTION_NARROWING": [
        domain("exception_water_gate", "水闸远程控制安全锁", "train", "exception_control", "安全锁定→紧迫性→即时风险与无替代→远程操作/现场接管", "远程打开水闸", "restricted_operation_allowed", "restricted_operation_blocked", "维持锁定并通知现场控制组", "记录风险判断和控制权交接", "restricted_condition", True),
        domain("exception_power_reserve", "储能系统备用模式切换", "train", "exception_energy", "备用模式限制→紧急需求→安全例外→切换/保留当前模式", "切换储能系统的备用供电模式", "emergency_switch_allowed", "switch_not_allowed", "保留当前模式并启用安全替代供电", "记录负载风险和切换日志", "restricted_condition", True),
        domain("exception_lab_shutdown", "实验室危险设备停机锁", "train", "exception_shutdown", "危险锁定→紧急停机需求→例外判断→停机/等待授权", "解除危险设备停机锁", "lock_release_allowed", "lock_remains", "保持停机锁并转现场安全负责人", "记录停机状态和授权过程", "restricted_condition", True),
        domain("exception_drone_launch", "应急测绘无人机禁飞区", "development", "exception_launch", "禁飞范围→任务紧迫性→安全替代→起飞/地面测绘", "在临时限制区域启动无人机", "emergency_launch_allowed", "launch_blocked", "改用地面测绘或等待安全许可", "记录飞行边界和地面替代方案", "restricted_condition", True),
        domain("exception_chemical_transfer", "受控化学品的紧急转移", "test", "exception_transfer", "转移限制→事故紧迫性→即时风险与替代条件→转移/封存", "转移受控化学品容器", "emergency_transfer_allowed", "transfer_blocked", "就地封存并呼叫具备资质的响应组", "记录容器状态和责任交接", "restricted_condition", True),
        domain("exception_route_barrier", "野外路线临时封控线", "test", "exception_route", "封控状态→通行请求→生命安全例外→穿越/改道", "穿越临时封控路段", "emergency_crossing_allowed", "crossing_blocked", "留在安全区域并使用替代路线", "记录人员位置与路线决定", "restricted_condition", True),
    ],
    "RTE_PRIORITY_UPPER_OVER_LOCAL": [
        domain("priority_export_queue", "跨项目材料导出队列", "train", "priority_export", "上层数据保护约束与本地赶工例外冲突→导出/脱敏摘要", "导出未完成核验的项目材料", "local_export_allowed", "export_blocked", "只提供已核验的脱敏摘要", "记录导出字段和审批理由", "direct_rule_conflict", False, "本地队列规则允许赶工导出，上层保护规则可禁止未核验材料离开项目区。"),
        domain("priority_machine_override", "自动设备的维护覆盖权限", "train", "priority_machine", "上层防护联锁与本地维护覆盖冲突→启动/保持联锁", "在联锁条件未满足时启动设备", "local_override_allowed", "upper_interlock_controls", "保持联锁并申请安全复位", "记录联锁状态和复位人", "direct_rule_conflict", False, "本地维护流程允许主管覆盖联锁，上层安全规则可要求联锁保持。"),
        domain("priority_record_release", "档案访问中的限制记录披露", "train", "priority_record", "上层保密约束与本地公共查阅规则冲突→披露/限制披露", "向申请人披露受限制记录", "local_disclosure_allowed", "disclosure_blocked", "仅披露不受限制的目录信息", "记录披露范围和限制依据", "direct_rule_conflict", False, "本地开放目录规则支持快速提供记录，上层保密条款可限制具体内容披露。"),
        domain("priority_vehicle_route", "维修车队的临时路线许可", "development", "priority_route", "上层危险区禁入与本地运维捷径冲突→通行/绕行", "驾驶维修车辆进入危险区捷径", "local_route_allowed", "upper_no_entry_applies", "改走已核验的安全路线", "记录路线选择和风险通知", "direct_rule_conflict", False, "本地运维时限规则允许使用捷径，上层危险区规则可禁止进入。"),
        domain("priority_account_restore", "服务账号的紧急恢复例外", "test", "priority_account", "上层身份核验要求与本地快速恢复规则冲突→恢复/暂缓", "恢复尚未核验身份的高权限账号", "local_restore_allowed", "restore_blocked", "先恢复只读访问并继续核验", "记录账号权限变更和核验人", "direct_rule_conflict", False, "本地连续服务规则允许快速恢复，上层身份安全规则要求核验后才恢复高权限。"),
        domain("priority_sample_release", "研究样本的跨站点转运", "test", "priority_sample", "上层样本链条约束与本地快速转运规则冲突→转运/封存", "转运缺少链条确认的研究样本", "local_transfer_allowed", "transfer_blocked", "原地封存并补齐交接链", "记录样本封签与交接链", "direct_rule_conflict", False, "本地实验时限规则允许快速转运，上层样本完整性规则可要求链条确认。"),
    ],
    "RTE_TRANSITION_PENDING_CASES": [
        domain("transition_license_queue", "设备许可申请队列", "train", "transition_license", "提交日期→待处理/已执行状态→培训与负责人条件→发证/补件", "签发设备许可", "license_issuance_ready", "license_waiting_for_update", "暂停签发并通知补齐新资质", "记录申请版本与签发时间", "safety_training_current+qualified_lead_assigned", False, "新资质标准要求当前有效的安全培训记录和一名具资质的现场负责人。"),
        domain("transition_data_request", "研究数据访问申请队列", "train", "transition_data", "申请提交日期→是否已开放→用途、字段与删除安排→开放/暂缓", "开放研究数据访问", "data_access_ready", "data_access_waiting_for_update", "暂缓开放并要求补充用途说明", "记录数据访问范围和时间", "purpose_statement_present+field_scope_defined+deletion_plan_present", False, "新标准要求提交具体研究目的、字段范围和删除计划。"),
        domain("transition_grant_award", "项目资助拨付队列", "train", "transition_grant", "申请日期→拨付是否执行→资金来源与计划签署→拨付/待补证", "拨付项目资助", "grant_disbursement_ready", "grant_disbursement_held", "暂缓拨付并按新标准补证", "记录拨付依据和执行状态", "independent_budget_source_provided+disbursement_plan_signed", False, "新标准要求提供一项独立预算来源证明及负责人签署的拨付计划。"),
        domain("transition_archive_loan", "特藏外借预约队列", "development", "transition_archive", "预约提交日期→实物是否已交付→保管环境与交接联系人→交付/等待", "交付预约的特藏材料", "archive_handover_ready", "archive_handover_held", "暂不交付并补充保管承诺", "记录预约与实物交接状态", "storage_environment_committed+handover_contact_named", False, "新标准要求接收方提交保管环境承诺和具名交接联系人。"),
        domain("transition_route_access", "路线通行审批队列", "test", "transition_route", "通行申请日期→是否已通行→风险告知与返程联络→批准/暂缓", "发放路线通行凭证", "route_pass_ready", "route_pass_held", "保留申请但先完成新的风险确认", "记录风险确认版本", "risk_notice_signed+return_contact_plan_present", False, "新标准要求签署路线风险告知并提交返程联络安排。"),
        domain("transition_equipment_release", "检修设备恢复使用队列", "test", "transition_equipment", "申请提交日期→是否已恢复运行→独立技师检查与签字→解锁/继续隔离", "解除设备检修隔离", "equipment_release_ready", "equipment_release_held", "维持隔离，等待新标准要求的第二技师检查与签字", "记录隔离解除时间和检查签字", "independent_tech_review_complete+second_signoff_present", False, "新标准要求由不同于维修执行人的第二名技师完成检查并签字。"),
    ],
}

# Additional held-out domains are authored as distinct downstream rules, not
# relabelings. Each adds a different action, local condition, fallback, or
# invariant while sharing its event's upper-rule mechanism.
EXTRA_TEST_DOMAINS: dict[str, list[dict[str, Any]]] = {
    "RTE_AGE_18_TO_20": [
        domain("age_drone_corridor", "无人机测绘的独立飞行许可", "test", "permit", "年龄资格→飞行安全认证→独立飞行/模拟训练", "独立操控测绘无人机", "independent_flight_authorized", "simulator_only", "转入模拟飞行并等待合格飞行员", "保留飞行计划和地理围栏记录", "drone_safety_certificate", True, "只有具备当前有效的飞行安全证书者才能独立操控无人机；新年龄标准生效前已签发且仍有效的独立飞行证可在到期前继续使用；未满足时只能在模拟器训练。", grandfathered_credential_name="独立飞行证"),
        domain("age_biospecimen_transfer", "研究样本跨站点转移的授权资格", "test", "authority", "年龄资格→样本处分权限→本人授权/共同授权", "批准研究样本跨站点转移", "self_transfer_authority", "coapproval_required", "暂停转移并取得研究协调员共同授权", "记录样本编号和授权范围", "coapproval_done", False, "未达到上层年龄资格的申请人须取得样本负责人共同授权后，才可批准跨站点转移。"),
        domain("age_response_commander", "应急响应队的现场指挥任命", "test", "role", "年龄资格→现场指挥训练→指挥员/副手", "担任现场响应指挥员", "incident_commander_assigned", "deputy_assigned", "安排为副手并由合格指挥员负责", "记录指挥交接和风险简报", "incident_command_training", False, "现场指挥员还须完成应急指挥训练；未完成者只能担任副手。"),
        domain("age_water_control", "水处理设施的阀门调度权限", "test", "dispatch", "年龄资格→工艺考试→直接调度/监督席位", "下达取水阀门调度指令", "direct_valve_dispatch", "dispatch_not_authorized", "转入监督席位或暂停调度", "保留每次阀门指令与操作日志", "water_process_exam", False, "直接调度须通过水处理工艺考试；未通过者只能在监督席位影子操作。"),
        domain("age_isotope_checkout", "实验室同位素源的带离许可", "test", "lending", "取用地点→年龄资格→现场使用/带离审批", "将同位素源带离指定实验室", "offsite_source_checkout", "onsite_use_only", "仅限指定实验室内使用并办理保管登记", "保留源项交接与归还记录", "source_handling_training", False, "现场使用不受带离资格限制；将同位素源带离实验室须满足年龄标准和源项管理要求。"),
        domain("age_submersible_control", "水下测绘载具的远程操控权限", "test", "remote", "年龄资格→水下操控认证→远控/现场监护", "远程控制水下测绘载具", "remote_submersible_control", "supervised_console_only", "改由现场监护员操作或暂缓任务", "记录遥控会话和失联处置", "submersible_operator_exam", False, "远程操控须通过水下载具操作考试；未满足时只可在现场监护员接管的控制台操作。"),
        domain("age_event_safety_lead", "大型活动的现场安全负责人任命", "test", "role", "年龄资格→活动安全培训→负责人/协助员", "担任活动现场安全负责人", "safety_lead_assigned", "safety_assistant_assigned", "安排为协助员并由合格负责人签核", "保留安全简报和岗位分工", "event_safety_training", False, "活动安全负责人须完成场地疏散与人群风险培训；未完成者只能协助，不得独立负责。"),
        domain("age_cargo_lift", "货运升降平台的独立操作许可", "test", "permit", "年龄资格→载荷设备证书→独立操作/训练模式", "独立操控货运升降平台", "lift_operation_authorized", "training_mode_only", "锁定独立模式并转入训练控制", "保留载荷核对和设备检查记录", "lift_operator_certificate", True, "独立操作还须具备货运升降平台证书；新年龄标准生效前已签发且仍有效的独立操作证可在到期前继续使用；证书缺失时只允许使用训练模式。", grandfathered_credential_name="货运升降平台独立操作证"),
    ],
    "RTE_BIOMETRIC_VOICE_EXPANSION": [
        domain("voice_lab_safety_command", "实验室安全联锁的声纹指令", "test", "biometric_dispatch", "声纹身份识别→指令授权→执行联锁/值班员核对权限", "依据声纹执行安全联锁指令", "voice_interlock_command", "manual_interlock_review", "由值班员核对指令权限后再执行", "记录指令来源和联锁状态", "separate_consent+encrypted+purpose_match+retention_valid", False, "安全联锁指令仅能用于登记的实验室用途，必须单独授权、加密且保存不超过30日。"),
        domain("voice_restricted_zone", "物流限制区的声纹门禁", "test", "biometric_entry", "声纹匹配→区域准入→声纹通行/实体凭证", "使用声纹进入限制区", "voice_zone_entry", "badge_only_entry", "改用实体凭证并由门岗核验", "保留区域通行和身份核验记录", "separate_consent+purpose_match+retention_valid", False, "声纹门禁须取得单独授权、用途与限制区准入相符，且识别记录在30日内删除。"),
        domain("voice_energy_profile", "能源门户的个性化用量检索", "test", "biometric_profile", "身份特征生成→个人化检索→个性化/通用查询", "按声纹档案展示个人用量", "personalized_usage_view", "generic_usage_view", "关闭个人化展示，保留通用查询", "记录访问目的和个性化配置", "separate_consent+purpose_match", False, "只有针对用量展示的单独授权且用途匹配时，才可按声纹档案个性化检索。"),
        domain("voice_maintenance_release", "维修控制台的声纹交接授权", "test", "biometric_release", "身份认证→维修权限交接→授权放行/一次性密钥", "确认维修人员并开放控制台", "voice_console_release", "temporary_key_required", "改用一次性密钥并由交接员核验", "保留设备交接与认证日志", "separate_consent+encrypted+purpose_match", False, "声纹交接须单独授权、加密保存且仅用于本次维修权限交接。"),
        domain("voice_archive_case_access", "档案调阅终端的声纹自助核验", "test", "biometric_records", "身份特征范围→档案访问等级→自助调阅/馆员核验", "自助调取限制级档案", "self_service_archive_access", "archivist_verification_required", "转由档案员核验证件和调阅权限", "记录档案编号与访问主体", "separate_consent+encrypted+purpose_match+retention_valid", False, "限制级档案的声纹自助调阅须单独授权、加密、用途匹配并在30日内删除模板。"),
        domain("voice_transcript_search", "维修工单语音转写检索", "test", "biometric_transcript", "语音转写→身份特征是否生成→内容检索/禁止身份匹配", "按转写内容检索维修工单", "transcript_search_allowed", "identity_matching_disabled", "保留内容检索但禁用说话者身份匹配", "记录检索字段和工单关联理由", "identity_inference+identity_feature_created", False, "只允许按转写内容检索；不得生成或使用说话者身份特征。"),
        domain("voice_fleet_dispatch", "车队维护任务的声纹指令调度", "test", "biometric_dispatch", "声纹身份识别→车辆任务权限→自动派发/调度员确认", "依据声纹派发维护车辆任务", "voice_fleet_dispatch", "dispatcher_confirmation_required", "由调度员核对任务权限后派发", "保留车辆任务和授权版本", "separate_consent+purpose_match+retention_valid", False, "语音身份调度须有单独授权、用途匹配且保存不超过30日；未满足时由调度员确认。"),
    ],
    "RTE_MEANINGFUL_HUMAN_REVIEW": [
        domain("review_vendor_contract", "关键供应合同的自动停约建议", "test", "review_vendor_contract", "替代供应是否可用→停约影响判定→合同暂停或审查", "暂停关键供应合同", "contract_pause_ready", "contract_pause_held", "维持临时供货并进入审查队列", "保留替代来源核验和合同通知记录", "no_qualified_substitute_available", False, "若没有已核验的替代供应方，自动暂停合同会中断关键服务并视为高影响；有合格替代来源时按一般合同流程处理。", impact_fact="no_qualified_substitute_available", impact_true_text="尚无经过核验的替代供应方", impact_false_text="已有一家具备交付资格的替代供应方"),
        domain("review_aid_payment", "资助款项的自动暂停拨付", "test", "review_aid_payment", "本期款项是否为基本生活所需→暂停影响判定→拨付或审查", "暂停本期资助款拨付", "payment_pause_ready", "payment_pause_held", "维持已核准的本期拨付并复核", "记录资金状态和受助人通知", "current_payment_is_livelihood", False, "若本期款项用于受助人基本生活支出，自动暂停拨付构成高影响；已安排等额替代拨付时不满足该影响条件。", impact_fact="current_payment_is_livelihood", impact_true_text="本期资助款用于受助人的基本生活支出", impact_false_text="受助人已有等额替代拨付安排"),
        domain("review_content_distribution", "内容平台的自动降权处理", "test", "review_content_distribution", "处理是否覆盖全部分发渠道→表达影响判定→降权或人工复核", "全面限制账户内容分发", "distribution_limit_ready", "distribution_limit_held", "仅标记争议内容并保留申诉入口", "保留内容版本和申诉记录", "all_channels_restricted", False, "限制账户全部分发渠道才构成高影响；仅降低单条内容排序时不触发高影响复核。", impact_fact="all_channels_restricted", impact_true_text="拟处理会限制账户在全部渠道的内容分发", impact_false_text="拟处理只降低一条内容的排序"),
        domain("review_equipment_write_access", "安全设备的自动写入权限撤销", "test", "review_equipment_write", "写入权限是否控制安全联锁→权限撤销影响判定→撤销或审查", "撤销安全设备写入权限", "write_access_revocation_ready", "write_access_revocation_held", "先限制高风险命令并等待技术审查", "保留联锁状态和权限变更日志", "write_controls_safety_interlock", False, "若写入权限可改变安全联锁参数，自动撤销构成高影响；只读监测权限不满足该条件。", impact_fact="write_controls_safety_interlock", impact_true_text="该写入权限可修改安全联锁参数", impact_false_text="该权限仅能读取监测信息"),
        domain("review_badge_shift_access", "夜班岗位门禁权限的自动停用", "test", "review_badge_shift", "夜班是否无替代入口→停权影响判定→停用或临时通行", "停用夜班岗位门禁权限", "badge_disable_ready", "badge_disable_held", "发放限时临时通行凭证并复核", "记录临时凭证范围和门禁事件", "no_alternative_night_entry", False, "若夜班岗位没有其他安全入口，自动停用门禁会阻断必要工作并构成高影响；存在替代入口时不触发该条件。", impact_fact="no_alternative_night_entry", impact_true_text="夜班岗位没有其他安全入口", impact_false_text="夜班人员可使用另一处经核验的安全入口"),
        domain("review_research_dosing", "研究干预剂量的自动暂停建议", "test", "review_research_dosing", "剂量是否处于连续干预周期→中断影响判定→暂停或安全复核", "暂停研究干预剂量安排", "dose_pause_ready", "dose_pause_held", "转由研究医师核实后再变更安排", "记录干预时间线和安全联系情况", "dose_is_in_active_cycle", False, "参与者正在连续干预周期内时，自动暂停剂量构成高影响；非干预随访阶段不满足该影响条件。", impact_fact="dose_is_in_active_cycle", impact_true_text="参与者处于连续干预周期内", impact_false_text="参与者当前处于非干预随访阶段"),
        domain("review_housing_allocation", "临时住宿资格的自动取消", "test", "review_housing", "是否无其他可用住所→取消影响判定→取消或暂缓", "取消临时住宿资格", "housing_cancel_ready", "housing_cancel_held", "维持短期住宿并核实替代安排", "记录住宿安排和通知送达情况", "no_safe_alternative_housing", False, "当事人没有安全可用的替代住所时，自动取消住宿资格构成高影响；已有可用替代住所时不满足该条件。", impact_fact="no_safe_alternative_housing", impact_true_text="当事人没有安全可用的替代住所", impact_false_text="当事人已有经核实的安全替代住所"),
        domain("review_license_registry", "专业许可登记的自动暂停", "test", "review_license", "暂停是否使关键服务资格中断→登记影响判定→暂停或审查", "暂停专业许可登记状态", "license_pause_ready", "license_pause_held", "保留只读核验状态并先完成人审", "记录登记状态和查询方通知", "license_needed_for_essential_service", False, "若许可是提供关键服务所必需的唯一资格，自动暂停登记构成高影响；仅暂停非关键扩展权限时不满足该条件。", impact_fact="license_needed_for_essential_service", impact_true_text="该许可是提供关键服务所必需的唯一资格", impact_false_text="被暂停的只是非关键扩展权限"),
    ],
    "RTE_VALIDITY_180_TO_90": [
        domain("valid_water_release", "饮用水处理批次的放行核验", "test", "validity_hold", "核验有效期→水质放行门槛→放行/重新取样", "放行处理水批次", "water_batch_release", "water_batch_held", "暂缓出厂并重新核验水质", "保留取样链和批次放行记录", "independent_water_sample_valid", False, "批次放行还须有独立水样和完整取样链；有效期内但缺少该条件仍不得放行。"),
        domain("valid_cold_chain", "冷链样本运输许可", "test", "validity_route", "路线核验时效→温控连续性→按原路线运输/替代路线", "沿核准路线运输冷链样本", "cold_chain_route_allowed", "cold_chain_route_deferred", "改走具有有效温控记录的路线", "记录运输温度和路线变更", "temperature_monitor_active", False, "路线核验有效时仍须有运行中的温度监测器；否则应改走已验证路线。"),
        domain("valid_research_identity", "研究账户身份认证续期", "test", "validity_downgrade", "身份核验时效→权限范围→全权限/只读", "以研究账户写入新记录", "research_write_allowed", "research_read_only", "降为只读直到重新核验身份", "保留账户授权和写入日志", "research_role_current", False, "当前身份核验有效且研究角色仍在任期内，方可写入记录；否则保留只读访问。"),
        domain("valid_pressure_vessel", "压力容器复检与恢复使用", "test", "validity_maintenance", "复检有效期→保护阀状态→恢复使用/停机检修", "启动压力容器", "vessel_start_allowed", "vessel_start_blocked", "保持停机并由技师复检保护阀", "保留复检数据和保护阀维护记录", "relief_valve_test_passed", False, "复检记录有效还不够，保护阀也必须通过本周期测试。"),
        domain("valid_archival_environment", "档案库房环境合规放行", "test", "validity_quarantine", "环境检测时效→湿度控制→开放存取/隔离材料", "开放档案库房存取", "archive_access_allowed", "archive_access_quarantined", "暂缓调取并隔离受影响材料", "保留环境曲线和材料搬移记录", "humidity_control_operational", False, "环境核验有效且湿度控制设备运行正常，方可继续开放库房。"),
        domain("valid_reactor_console", "实验反应装置控制台授权", "test", "validity_downgrade", "授权时效→双人值守→完整控制/只读监控", "远程修改反应装置参数", "parameter_write_allowed", "monitor_only", "降为只读监控并安排合格人员重新检查设备", "记录参数修改人和控制台访问", "second_operator_on_shift", False, "身份核验在有效期内且有第二名合格操作员同时值守，才允许远程修改参数。"),
        domain("valid_survey_permit", "野外取样许可续期", "test", "validity_route", "风险评估时效→天气窗口→按许可取样/改点", "进入指定地点开展取样", "sampling_entry_allowed", "sampling_entry_deferred", "改到已核验地点或等待新的风险评估", "记录取样坐标和天气窗口", "weather_window_safe", False, "风险评估有效且天气窗口安全，方可进入指定取样点。"),
        domain("valid_robot_firmware", "仓储机器人固件部署资格", "test", "validity_hold", "安全核验时效→回滚包可用性→部署/暂缓", "向仓储机器人部署新固件", "firmware_deployment_allowed", "firmware_deployment_held", "暂缓部署并重新执行安全核验", "保留版本哈希和回滚记录", "rollback_image_available", False, "核验有效且有可用回滚镜像，才可部署新固件。"),
    ],
    "RTE_EVIDENCE_1_TO_2": [
        domain("evidence_water_contamination", "水样污染源认定", "test", "evidence_provenance", "独立来源数量→污染源归属→确认/继续取证", "确认污染源并发布处置通知", "source_attribution_supported", "source_attribution_unresolved", "维持临时防护并补充独立检测", "记录采样点和检测链", "sampling_site_matches_notice", False, "采样地点必须与拟发布的处置通知覆盖地点一致。", secondary_gate_label="采样地点与通知覆盖范围一致"),
        domain("evidence_firmware_fault", "自动设备故障归因", "test", "evidence_diagnosis", "独立诊断来源→故障重现→更换部件/继续观察", "认定固件缺陷并部署修复", "firmware_fault_supported", "firmware_fault_unverified", "暂不部署修复并进行隔离测试", "记录复现版本和测试环境", "fault_reproduced_under_same_version", False, "故障须在相同固件版本和隔离环境中复现后才可归因。", secondary_gate_label="故障在相同版本的隔离环境中复现"),
        domain("evidence_funding_origin", "项目资金来源合规核验", "test", "evidence_eligibility", "独立资金来源→限制用途检查→确认/冻结拨付", "确认项目资金来源符合要求", "funding_origin_supported", "funding_origin_unverified", "暂缓拨付并核对资金用途限制", "记录资金来源和限制条款", "funds_not_restricted_for_project", False, "资金来源须允许本项目用途，否则即使来源证据充分也不得确认。", secondary_gate_label="资金未被限制用于其他项目"),
        domain("evidence_archive_chain", "档案移交链完整性核验", "test", "evidence_confirmation", "独立移交记录→保管链连续性→开放调阅/暂缓", "确认档案材料可供调阅", "archive_chain_confirmed", "archive_chain_incomplete", "暂停公开并补齐交接记录", "保留原始交接编号", "custody_chain_has_no_gap", False, "保管链不能有未解释的交接空档。", secondary_gate_label="档案保管链没有未解释的空档"),
        domain("evidence_sample_integrity", "研究样本质量异常确认", "test", "evidence_clearance", "独立检测来源→样本完整性→解封/继续隔离", "解除样本质量隔离", "sample_release_supported", "sample_remains_isolated", "维持隔离并申请独立复测", "记录样本封签与检测批次", "sample_seal_intact", False, "样本封签完整且编号匹配，方可解除隔离。", secondary_gate_label="样本封签完整且编号匹配"),
        domain("evidence_resource_claim", "设备资源申领资格核验", "test", "evidence_claim", "独立申请记录→申请人身份匹配→分配/补充核验", "确认设备资源申领并安排使用", "resource_claim_supported", "resource_claim_unresolved", "暂缓分配并核对申请身份", "记录分配批次和申请人", "requester_identity_matches", False, "申领记录必须与实际申请人身份匹配，才可安排资源。", secondary_gate_label="申请记录与实际申请人身份匹配"),
        domain("evidence_repair_clearance", "维修完成后的安全复机认定", "test", "evidence_diagnosis", "独立维修证据→负载测试→复机/继续锁定", "解除设备维修锁定", "repair_clearance_supported", "repair_clearance_pending", "保持锁定并完成独立负载测试", "保留维修工单和复机授权", "post_repair_load_test_passed", False, "维修完成后须通过独立负载测试才可复机。", secondary_gate_label="维修后独立负载测试通过"),
    ],
    "RTE_DEFAULT_TO_OPT_IN": [
        domain("optin_facility_telemetry", "设施能耗数据的跨机构共享", "test", "permission_sharing", "共享授权→接收方范围→共享/本地留存", "向合作机构共享设施能耗数据", "facility_data_share_allowed", "facility_data_local_only", "数据留在本机构并只提供汇总值", "记录接收方和字段范围", "affirmative_authorization", False, "跨机构共享须有针对该接收方和能耗用途的肯定授权。"),
        domain("optin_model_validation", "模型验证数据的二次训练授权", "test", "permission_training", "训练用途授权→样本范围→加入验证训练集/排除", "将历史样本用于模型验证训练", "validation_training_allowed", "validation_training_blocked", "将样本隔离在训练集之外", "记录样本来源和授权版本", "affirmative_authorization", False, "模型验证训练属于新的二次用途，须取得明确肯定授权。"),
        domain("optin_repair_followup", "维修完成后的客户回访", "test", "permission_recontact", "回访授权→联系渠道→发送回访/停止联系", "发送维修满意度回访", "repair_followup_allowed", "repair_followup_blocked", "不发送营销或满意度信息，只保留必要服务通知", "记录联系渠道和目的", "affirmative_authorization", False, "满意度回访须有针对该目的的肯定授权；服务通知不扩大为回访许可。"),
        domain("optin_worker_analytics", "工作区行为统计分析", "test", "permission_analytics", "分析授权→数据粒度→统计分析/仅保留安全记录", "将工作区行为记录用于效率分析", "workplace_analysis_allowed", "security_logs_only", "停止效率分析，仅保留必要安全记录", "记录分析字段和访问角色", "affirmative_authorization", False, "效率分析属于二次用途，未获肯定授权时仅可保留安全事件记录。"),
        domain("optin_archive_search", "档案影像的跨目录语义索引", "test", "permission_indexing", "索引授权→跨目录可见范围→建立索引/保持未索引", "将档案影像加入跨目录搜索索引", "archive_indexing_allowed", "archive_not_indexed", "不建立跨目录索引，保持原目录管理", "记录索引范围和移除状态", "affirmative_authorization", False, "跨目录索引须获得明确授权，未回应不能视为同意。"),
        domain("optin_equipment_benchmark", "设备遥测数据的外部基准共享", "test", "permission_sharing", "对外共享授权→字段脱敏→共享/仅内部统计", "向外部基准项目提供设备遥测数据", "benchmark_share_allowed", "internal_statistics_only", "仅在本项目内部汇总，不提供外部明细", "记录字段脱敏和共享对象", "affirmative_authorization", False, "对外提供遥测数据须有明确授权；本地汇总不代表获准披露明细。"),
        domain("optin_audio_transcript", "服务录音转写的研究语料复用", "test", "permission_reuse", "语料复用授权→用途匹配→纳入研究语料/仅用于本次服务", "将服务转写文本用于研究语料", "transcript_reuse_allowed", "service_transcript_only", "保留原服务记录但不进入研究语料", "记录语料用途和删除期限", "affirmative_authorization", False, "服务转写文本用于研究语料属于二次用途，须单独明确授权。"),
    ],
    "RTE_EMERGENCY_EXCEPTION_NARROWING": [
        domain("exception_tunnel_ventilation", "隧道通风系统的紧急旁路", "test", "exception_control", "旁路限制→紧迫性→即时人身风险与无替代→开启旁路/现场接管", "开启隧道通风系统旁路", "ventilation_bypass_allowed", "ventilation_bypass_blocked", "保持自动控制并派现场工程组", "记录风险判断和控制权交接", "restricted_condition", True, "本地规则通常禁止旁路；需要即时人身风险且无安全替代时才可进入紧急通道。"),
        domain("exception_battery_isolation", "储能电池组的紧急隔离解除", "test", "exception_energy", "隔离锁→事故紧迫性→即时风险与替代方案→解除/维持隔离", "解除储能电池组隔离", "battery_isolation_release_allowed", "battery_isolation_maintained", "维持隔离并启用独立供电", "记录电池温度和现场响应", "restricted_condition", True, "隔离解除通常禁止；即时人身安全风险且无安全替代供电时才可例外。"),
        domain("exception_cleanroom_lock", "洁净室污染事件的门禁锁定", "test", "exception_shutdown", "人员进入限制→紧急需求→即时安全风险与无替代→开锁/保持封闭", "解除洁净室人员进入锁定", "cleanroom_entry_exception_allowed", "cleanroom_lock_remains", "保持封闭并由外部处置组处理", "记录污染风险和进出人员", "restricted_condition", True, "洁净室封闭状态通常不得解除；只有存在即时人身危险且无安全替代路径时才例外。"),
        domain("exception_airspace_inspection", "临时管制空域的巡检无人机起飞", "test", "exception_launch", "空域限制→任务紧迫性→即时人身风险与无替代→起飞/地面巡检", "在临时管制空域启动巡检无人机", "airspace_exception_allowed", "drone_launch_blocked", "改用地面巡检并通知空域协调员", "记录飞行边界和地面替代方案", "restricted_condition", True, "空域限制期间通常不得起飞；即时人身安全风险且无安全替代巡检方式时才例外。"),
        domain("exception_reagent_transfer", "危险试剂的紧急容器转移", "test", "exception_transfer", "容器转移限制→泄漏紧迫性→即时风险与无替代→转移/原地封控", "转移危险试剂容器", "reagent_transfer_exception_allowed", "reagent_transfer_blocked", "原地封控并呼叫合格处置人员", "记录容器状态和交接责任", "restricted_condition", True, "危险试剂通常不得转移；容器正在造成即时人身危险且无安全封控替代时才例外。"),
        domain("exception_bridge_barrier", "桥梁结构风险区域的封闭隔离线", "test", "exception_route", "封闭区通行限制→人员请求→即时生命风险与无替代→穿越/改道", "穿越桥梁风险区域的隔离线", "bridge_barrier_exception_allowed", "barrier_crossing_blocked", "留在安全侧并联系救援队", "记录人员位置和路线决定", "restricted_condition", True, "结构风险区域通常不得穿越；只有即时生命危险且没有安全替代路径时才适用例外。"),
        domain("exception_waste_neutralization", "危险废液中和流程的紧急暂停解除", "test", "exception_shutdown", "中和流程锁定→紧迫性→即时人员风险与无替代→解除/继续锁定", "解除危险废液中和流程锁定", "neutralization_lock_release_allowed", "neutralization_lock_maintained", "保持流程锁定并启动外部安全处置", "记录化学品状态和处置授权", "restricted_condition", True, "中和流程锁定一般不得解除；只有即时人身风险且无安全替代处置时才例外。"),
    ],
    "RTE_PRIORITY_UPPER_OVER_LOCAL": [
        domain("priority_lab_export", "实验数据导出的脱敏优先级", "test", "priority_export", "本地协作规则与上层隐私限制冲突→原始导出/脱敏摘要", "导出含未核验身份字段的实验数据", "raw_export_allowed", "raw_export_blocked", "仅导出已核验的聚合摘要", "记录字段清单和脱敏版本", "direct_rule_conflict", False, "本地协作流程允许导出原始记录；上层隐私限制禁止未核验身份字段离开受控区。"),
        domain("priority_pressure_lock", "压力设备维护模式下的联锁优先级", "test", "priority_machine", "维护覆盖指令与上层联锁冲突→启动/维持保护联锁", "在联锁未满足时启动压力设备", "maintenance_override_allowed", "upper_interlock_controls", "保持联锁并申请双人复位", "记录联锁输入和复位人员", "direct_rule_conflict", False, "本地维护手册允许主管临时覆盖；上层压力保护规则要求联锁条件满足后才能启动。"),
        domain("priority_archive_release", "限制档案申请的披露范围", "test", "priority_record", "公共查阅请求与上层保密边界冲突→披露/目录级答复", "披露尚未完成保密核验的档案内容", "full_disclosure_allowed", "restricted_disclosure_blocked", "仅提供不含限制内容的目录信息", "记录披露依据和限制条款", "direct_rule_conflict", False, "本地快速查阅规则支持提供申请内容；上层保密限制要求完成核验后才可披露正文。"),
        domain("priority_vehicle_hazard", "维护车辆进入危险区的路线优先级", "test", "priority_route", "本地捷径授权与上层危险区限制冲突→进入/安全绕行", "驾驶维护车辆进入上层禁入危险区", "local_shortcut_allowed", "upper_no_entry_applies", "改走经核验的外围路线", "记录路线和危险区告知", "direct_rule_conflict", False, "本地运维时限允许走捷径；上层危险区规则禁止进入仍受限制的区域。"),
        domain("priority_identity_restore", "高权限账户的快速恢复与身份核验", "test", "priority_account", "本地连续服务例外与上层身份控制冲突→恢复高权限/只读", "恢复尚未完成核验的高权限账户", "rapid_restore_allowed", "privileged_restore_blocked", "只恢复只读访问并继续身份核验", "记录权限级别和核验进度", "direct_rule_conflict", False, "本地业务连续性规则允许快速恢复；上层身份安全规则要求完成核验后才能恢复写入权限。"),
        domain("priority_sample_custody", "研究样本快速转运与保管链优先级", "test", "priority_sample", "本地时限例外与上层保管链要求冲突→转运/原地封存", "转运交接链尚未确认的研究样本", "rapid_transfer_allowed", "transfer_blocked", "原地封存并补齐交接记录", "记录封签和每次责任移交", "direct_rule_conflict", False, "本地时限规则允许快速转运；上层样本完整性规则要求交接链确认后才可转运。"),
        domain("priority_water_bypass", "供水事故中的净化旁路权限", "test", "priority_machine", "本地供水连续性例外与上层净化安全条件冲突→旁路/保持净化", "绕过尚未满足的净化联锁", "local_bypass_allowed", "upper_purification_controls", "保持净化联锁并启用备用水源", "记录水质参数和切换时间", "direct_rule_conflict", False, "本地连续供水例外允许临时旁路；上层水质规则要求净化联锁满足后才能供水。"),
    ],
    "RTE_TRANSITION_PENDING_CASES": [
        domain("transition_lab_access", "高风险实验区的门禁申请队列", "test", "transition_license", "提交日期→门禁是否已开通→安全培训和现场负责人→开通/暂缓", "开通高风险实验区门禁", "lab_access_ready", "lab_access_held", "保留申请并补齐培训记录", "记录门禁版本和授权时间", "hazard_training_current+qualified_shift_lead_assigned", False, "新标准要求有效的危险品培训记录和具资质的当班负责人。", secondary_gate_label="危险品培训在有效期内+具资质的当班负责人已安排"),
        domain("transition_data_export", "研究资料导出审批队列", "test", "transition_data", "申请日期→文件是否已导出→字段清单和删除安排→导出/暂缓", "批准研究资料导出", "research_export_ready", "research_export_held", "暂缓导出并要求补齐材料", "记录导出版本和字段范围", "export_field_manifest_present+deletion_deadline_defined", False, "新标准要求列明导出字段并规定可执行的删除期限。", secondary_gate_label="导出字段清单已列明+删除期限已规定"),
        domain("transition_equipment_license", "移动设备操作许可证签发队列", "test", "transition_license", "申领日期→许可证是否已签发→风险培训和责任人→签发/补件", "签发移动设备操作许可证", "mobile_license_ready", "mobile_license_held", "保留申请并补齐新资质", "记录许可证版本和签发责任人", "mobile_safety_training_current+accountable_operator_named", False, "新标准要求移动设备安全培训有效且具名责任人已确认。", secondary_gate_label="移动设备安全培训有效+具名操作责任人已确认"),
        domain("transition_archive_access", "特殊档案远程调阅申请队列", "test", "transition_data", "申请日期→远程权限是否已开通→用途说明和访问边界→开通/暂缓", "开通特殊档案远程调阅", "remote_archive_access_ready", "remote_archive_access_held", "暂缓开通并补充访问范围", "记录访问范围和结束日期", "research_purpose_defined+remote_scope_limited", False, "新标准要求说明具体研究目的并限制远程访问范围。", secondary_gate_label="研究目的已说明+远程访问范围已限制"),
        domain("transition_transport_permit", "危险材料运输许可审批队列", "test", "transition_route", "申领日期→材料是否已发运→承运培训和应急联系人→批准/暂缓", "签发危险材料运输许可", "hazmat_transport_ready", "hazmat_transport_held", "暂停发运并补齐运输计划", "记录包装状态和承运链", "hazmat_training_current+emergency_contact_verified", False, "新标准要求运输培训在有效期内并核实应急联系人。", secondary_gate_label="运输培训有效+应急联系人已核实"),
        domain("transition_research_payment", "研究参与补偿发放队列", "test", "transition_grant", "申请日期→补偿是否已发放→金额依据和审批签字→发放/暂缓", "发放研究参与补偿", "participant_payment_ready", "participant_payment_held", "暂缓发放并完成新标准核对", "记录金额计算和审批版本", "payment_basis_documented+independent_approval_signed", False, "新标准要求记录补偿金额依据并由独立审批人签字。", secondary_gate_label="补偿金额依据已记录+独立审批人已签字"),
        domain("transition_sensor_deployment", "远程传感器布设申请队列", "test", "transition_license", "申请日期→传感器是否已启用→位置核验和维护计划→启用/暂缓", "启用远程监测传感器", "sensor_deployment_ready", "sensor_deployment_held", "保留申请但先完成位置与维护核验", "记录设备位置和校准周期", "site_location_verified+maintenance_cycle_defined", False, "新标准要求核实安装位置并规定维护周期。", secondary_gate_label="安装位置已核实+维护周期已规定"),
        domain("transition_resource_allocation", "共享设备资源分配队列", "test", "transition_grant", "申请日期→资源是否已分配→优先级依据和冲突声明→分配/暂缓", "批准共享设备资源分配", "shared_resource_ready", "shared_resource_held", "暂缓最终分配并补齐优先级资料", "记录申请顺序和冲突声明", "allocation_priority_documented+conflict_declaration_complete", False, "新标准要求说明分配优先级并完成利益冲突声明。", secondary_gate_label="分配优先级已说明+利益冲突声明已完成"),
    ],
}

for _event_id, _profiles in EXTRA_TEST_DOMAINS.items():
    DOMAINS[_event_id].extend(_profiles)

# Keep the age-event exception outcome evaluable across held-out downstream
# domains, not only in the three domains that introduced the exception. Each
# exception is domain-specific and preserves existing task-specific gates.
AGE_TEST_EXCEPTION_SPECS = {
    "age_remote_prototype": ("远程原型操作授权", "新年龄标准生效前签发且仍有效的远程原型操作授权，可在当前试验周期结束前继续使用；不免除远程控制资质条件。"),
    "age_biospecimen_transfer": ("跨站点样本转移批件", "新年龄标准生效前已批准的跨站点转移任务，可在本批样本交接完成前维持原授权；新任务仍适用新标准。"),
    "age_response_commander": ("现场指挥任命", "新年龄标准生效前已任命且任期未结束的现场指挥员，可完成已开始行动的剩余阶段；不免除应急指挥训练条件。"),
    "age_water_control": ("水处理调度权限", "新年龄标准生效前已授予的水处理系统调度权限，可在当前班次结束前继续使用；不免除工艺考试条件。"),
    "age_isotope_checkout": ("同位素外借单", "新年龄标准生效前已批准的源项外借单，可在登记的归还期限内继续有效；不免除源项管理条件。"),
    "age_submersible_control": ("水下载具远程操控许可", "新年龄标准生效前签发且仍有效的远程操控许可，可在当前连续任务结束前使用；不免除操作考试条件。"),
    "age_event_safety_lead": ("活动安全负责人任命", "新年龄标准生效前已确认的活动安全负责人，可完成已经开始活动的剩余时段；不免除场地疏散与人群风险培训条件。"),
}
for _profile in DOMAINS["RTE_AGE_18_TO_20"]:
    if _profile["split"] == "test" and not _profile["has_downstream_exception"]:
        _credential, _exception_text = AGE_TEST_EXCEPTION_SPECS[_profile["domain_id"]]
        _profile["has_downstream_exception"] = True
        _profile["grandfathered_credential_name"] = _credential
        _profile["local_rule"] = " ".join(x for x in (_profile["local_rule"], _exception_text) if x)

HISTORIC_DOMAINS = {
    "rental", "medical_consent", "platform_payout", "consumer_credit",
    "finance_voice_auth", "smart_home_access", "telehealth_identity",
    "workplace_voice_access", "credit_decision", "recruitment", "admissions",
    "public_benefits",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_seed(domain_id: str) -> int:
    return int(sha256_bytes(f"{SEED}:{domain_id}".encode())[:8], 16)


@lru_cache(maxsize=None)
def pattern_order(event_id: str, domain_id: str, factor: str) -> tuple[int, ...]:
    """A reproducible, domain-specific stratified ordering of 20 cases."""
    order = list(range(N_FAMILIES_PER_DOMAIN))
    random.Random(stable_seed(f"{event_id}:{domain_id}:{factor}")).shuffle(order)
    return tuple(order)


def pattern_rank(index: int, event: dict[str, Any], dom: dict[str, Any], factor: str) -> int:
    return pattern_order(event["event_id"], dom["domain_id"], factor)[index]


def pattern_code(index: int, event: dict[str, Any], dom: dict[str, Any], factor: str, modulus: int) -> int:
    return pattern_rank(index, event, dom, factor) % modulus


def pattern_value(index: int, event: dict[str, Any], dom: dict[str, Any], factor: str,
                  values: list[Any]) -> Any:
    return values[pattern_rank(index, event, dom, factor) % len(values)]


@lru_cache(maxsize=None)
def event_domain_offsets(event_id: str, domain_id: str, factor: str) -> tuple[int, ...]:
    rng = random.Random(stable_seed(f"{event_id}:{domain_id}:{factor}"))
    if factor == "action_date":
        # Keep exact transition-boundary cases and independently sample the
        # remaining dates on both sides for each domain.
        offsets = [-1, 0, 1]
        offsets += rng.sample(range(-365, -1), 8)
        offsets += rng.sample(range(2, 366), 9)
    elif factor == "filed_date":
        offsets = [-1, 0, 1]
        offsets += rng.sample(range(-180, -1), 8)
        offsets += rng.sample(range(2, 181), 9)
    else:
        raise ValueError(f"Unknown date-factor: {factor}")
    rng.shuffle(offsets)
    return tuple(offsets)


@lru_cache(maxsize=None)
def event_action_offsets(event_id: str) -> dict[tuple[str, int], int]:
    """Assign globally unique action dates within an event, balanced around its update."""
    domain_rows = sorted(DOMAINS[event_id], key=lambda d: d["domain_id"])
    slots = [(d["domain_id"], i) for d in domain_rows for i in range(N_FAMILIES_PER_DOMAIN)]
    rng = random.Random(stable_seed(f"{event_id}:all_action_dates"))
    rng.shuffle(slots)
    total = len(slots)
    old_total = round(total * 0.45)
    exception_slots = {
        (d["domain_id"], i)
        for d in domain_rows if d["split"] in {"train", "test"} and d["has_downstream_exception"]
        for i in range(3 if d["split"] == "test" else 2)
    }
    mapping: dict[tuple[str, int], int] = {}
    if exception_slots:
        exception_offsets = rng.sample(range(2, 731), len(exception_slots))
        mapping.update(zip(sorted(exception_slots), exception_offsets))
    available_slots = [slot for slot in slots if slot not in exception_slots]
    fixed = {-1: available_slots[0], 0: available_slots[1], 1: available_slots[2]}
    mapping.update({slot: offset for offset, slot in fixed.items()})
    remaining = [slot for slot in slots if slot not in mapping]
    rng.shuffle(remaining)
    old_remaining = old_total - 1  # the -1 boundary date is pre-effective
    new_remaining = len(remaining) - old_remaining  # includes the 0 and +1 boundary slots
    negative = rng.sample(range(-730, -1), old_remaining)
    positive_pool = [x for x in range(2, 731) if x not in set(mapping.values())]
    positive = rng.sample(positive_pool, new_remaining)
    for slot, offset in zip(remaining[:old_remaining], negative):
        mapping[slot] = offset
    for slot, offset in zip(remaining[old_remaining:], positive):
        mapping[slot] = offset
    if len(mapping) != total or len(set(mapping.values())) != total:
        raise AssertionError(f"action_date_assignment_not_unique:{event_id}")
    return mapping


def day_for(index: int, effective: str, event: dict[str, Any], dom: dict[str, Any]) -> str:
    base = date.fromisoformat(effective)
    offset = event_action_offsets(event["event_id"])[(dom["domain_id"], index)]
    return (base + timedelta(days=offset)).isoformat()


def age_at(birth: str, at: str) -> int:
    b, a = date.fromisoformat(birth), date.fromisoformat(at)
    return a.year - b.year - ((a.month, a.day) < (b.month, b.day))


def birth_for_age(age: int, action_date: str, near_birthday: bool) -> str:
    a = date.fromisoformat(action_date)
    if near_birthday:
        # Exactly one day before the next birthday, while retaining the
        # requested completed-year age.  Use the year of the next birthday
        # (which may differ when the action date is Dec 31).
        birthday = a + timedelta(days=1)
        birth_year = birthday.year - age - 1
        try:
            birth = date(birth_year, birthday.month, birthday.day)
        except ValueError:
            # A Feb-29 birthday in a non-leap birth year is represented as
            # Mar-1, which is still one day after a Feb-28 action date.
            if (birthday.month, birthday.day) != (2, 29): raise
            birth = date(birth_year, 3, 1)
    else:
        birth_year = a.year - age
        try:
            birth = date(birth_year, a.month, a.day)
        except ValueError:
            # On a Feb-29 action date, use Feb-28 in a non-leap birth year;
            # age_at then yields the requested completed-year age.
            if (a.month, a.day) != (2, 29): raise
            birth = date(birth_year, 2, 28)
    assert age_at(birth.isoformat(), action_date) == age
    return birth.isoformat()


def generate_facts(event: dict[str, Any], dom: dict[str, Any], index: int, rng: random.Random) -> dict[str, Any]:
    event_id, kind = event["event_id"], event["kind"]
    action_date = day_for(index, event["effective_date"], event, dom)
    exception_probe = (kind == "age_threshold" and dom["has_downstream_exception"]
                       and index < (3 if dom["split"] == "test" else 2 if dom["split"] == "train" else 0))
    f: dict[str, Any] = {"action_date": action_date}
    if kind == "age_threshold":
        ages = [17, 18, 19, 20, 21, 19, 18, 22, 16, 20, 19, 17, 21, 18, 23, 19, 20, 17, 18, 21]
        age = (18 + pattern_code(index, event, dom, "exception_probe_age", 2)) if exception_probe else pattern_value(index, event, dom, "age", ages)
        f["birth_date"] = birth_for_age(age, action_date, pattern_code(index, event, dom, "near_birthday", 2) == 1)
        mode = dom["mode"]
        if mode in {"permit", "dispatch", "remote"}: f["secondary_gate_met"] = pattern_code(index, event, dom, "secondary_gate", 4) != 0
        if mode in {"dispatch", "remote"}: f["supervision_available"] = pattern_code(index, event, dom, "supervision_available", 3) == 0
        if mode == "authority": f["coapproval_done"] = pattern_code(index, event, dom, "coapproval_done", 5) in {0, 2}
        if mode == "lending": f["request_mode"] = "onsite" if pattern_code(index, event, dom, "request_mode", 4) == 0 else "offsite"
        if dom["has_downstream_exception"]:
            f["grandfathered_credential"] = True if exception_probe else pattern_code(index, event, dom, "grandfathered_credential", 5) == 0
            if exception_probe and mode in {"permit", "dispatch", "remote"}:
                f["secondary_gate_met"] = True
            if exception_probe and mode == "role":
                f["field_training_complete"] = True
        if mode == "role": f["field_training_complete"] = pattern_code(index, event, dom, "field_training_complete", 3) != 1
    elif kind == "scope_expansion":
        is_transcript = dom["mode"] == "biometric_transcript"
        inference = False if is_transcript else pattern_code(index, event, dom, "identity_inference", 4) != 0
        f.update({"feature_type": "speech_transcript" if is_transcript else "voiceprint",
                  "identity_inference": inference,
                  "identity_feature_generated": False if is_transcript else inference and pattern_code(index, event, dom, "identity_feature_generated", 3) != 1})
        in_scope_after_update = (not is_transcript and f["feature_type"] == event["mechanism"]["added_feature"]
                                 and f["identity_inference"] and f["identity_feature_generated"]
                                 and date.fromisoformat(action_date) >= date.fromisoformat(event["effective_date"]))
        if in_scope_after_update:
            control_facts = {"separate_consent": pattern_code(index, event, dom, "separate_consent", 5) in {0, 2},
                             "encrypted": pattern_code(index, event, dom, "encrypted", 3) != 0,
                             "purpose_match": pattern_code(index, event, dom, "purpose_match", 4) != 2}
            for control in dom["secondary_gate"].split("+"):
                if control in control_facts:
                    key = "purpose_matches_domain" if control == "purpose_match" else control
                    f[key] = control_facts[control]
                elif control == "retention_valid":
                    f["retention_days"] = pattern_value(index, event, dom, "retention_days",
                                                        [14, 30, 31, 60, 10, 7, 45, 21, 28, 120,
                                                         9, 15, 32, 5, 60, 30, 14, 31, 25, 3])
    elif kind == "review_strengthening":
        impact_key = dom["impact_fact"] or "high_impact"
        impact = pattern_code(index, event, dom, "impact", 4) != 0
        adverse = pattern_code(index, event, dom, "adverse_proposed", 5) != 0
        f.update({impact_key: impact, "adverse_proposed": adverse})
        if impact and adverse:
            f.update({"formal_confirmation": pattern_code(index, event, dom, "formal_confirmation", 2) == 0,
                      "materials_checked": pattern_code(index, event, dom, "materials_checked", 3) != 0,
                      "system_reason_checked": pattern_code(index, event, dom, "system_reason_checked", 4) != 1,
                      "reviewer_can_override": pattern_code(index, event, dom, "reviewer_can_override", 5) != 2,
                      "review_reason_recorded": pattern_code(index, event, dom, "review_reason_recorded", 3) == 0,
                      "domain_file_access": pattern_code(index, event, dom, "domain_file_access", 4) != 3})
    elif kind == "validity_window_shortening":
        f.update({"days_since_last_check": pattern_value(index, event, dom, "days_since_last_check",
                                                         [30, 89, 90, 91, 120, 179, 180, 181, 45, 150,
                                                          1, 60, 88, 92, 100, 170, 182, 15, 95, 240]),
                  "secondary_gate_met": pattern_code(index, event, dom, "secondary_gate", 3) != 1})
    elif kind == "evidence_threshold_increase":
        source_counts = [0, 1, 2, 3, 1, 2, 0, 3, 1, 2, 3, 0, 1, 2, 3, 0, 2, 1, 3, 2]
        source_count = pattern_value(index, event, dom, "independent_source_count", source_counts)
        f.update({"independent_source_count": source_count,
                  "core_claim_true": pattern_code(index, event, dom, "core_claim_true", 5) != 4,
                  "evidence_sources": [f"独立来源{j+1}" for j in range(source_count)]})
        f[dom["secondary_gate"]] = pattern_code(index, event, dom, "secondary_gate", 4) != 1
    elif kind == "permission_default_reversal":
        record_exists = pattern_code(index, event, dom, "authorization_record_exists", 4) != 1
        scope_matches = not record_exists or pattern_code(index, event, dom, "authorization_scope_matches", 3) != 0
        objection = pattern_code(index, event, dom, "explicit_objection", 5) == 3
        affirmative = (record_exists and scope_matches and not objection
                       and pattern_code(index, event, dom, "affirmative_authorization", 5) in {0, 1})
        f.update({"authorization_record_exists": record_exists,
                  "affirmative_authorization": affirmative,
                  "explicit_objection": objection,
                  "response_received": affirmative or objection or pattern_code(index, event, dom, "response_received", 4) == 3,
                  "authorization_scope_matches": scope_matches})
    elif kind == "exception_narrowing":
        immediate_risk = pattern_code(index, event, dom, "immediate_safety_risk", 4) == 0
        f.update({"restricted_condition": pattern_code(index, event, dom, "restricted_condition", 5) != 0,
                  "urgent": (pattern_code(index, event, dom, "urgent", 3) != 1) or immediate_risk,
                  "immediate_safety_risk": immediate_risk,
                  "safe_alternative_available": pattern_code(index, event, dom, "safe_alternative_available", 4) == 1,
                  "qualified_responder_available": pattern_code(index, event, dom, "qualified_responder_available", 3) == 0})
    elif kind == "conflict_priority_change":
        upper_applies = pattern_code(index, event, dom, "upper_safety_restriction_applies", 3) != 0
        local_allows = pattern_code(index, event, dom, "local_exception_allows_action", 4) != 2
        f.update({"upper_safety_restriction_applies": upper_applies,
                  "local_exception_allows_action": local_allows,
                  "direct_conflict": upper_applies and local_allows and pattern_code(index, event, dom, "direct_conflict", 5) != 0,
                  "independent_safeguard_complete": pattern_code(index, event, dom, "independent_safeguard_complete", 3) == 1})
    elif kind == "transition_scope_expansion":
        offsets = event_domain_offsets(event_id, dom["domain_id"], "filed_date")
        action_offset = (date.fromisoformat(action_date) - date.fromisoformat(event["effective_date"])).days
        filed_offset = min(offsets[index], action_offset)
        filed_date = date.fromisoformat(event["effective_date"]) + timedelta(days=filed_offset)
        request_status = "pending" if pattern_code(index, event, dom, "request_status", 4) != 0 else "executed"
        f.update({"filed_date": filed_date.isoformat(), "request_status": request_status})
        if request_status == "pending":
            if filed_date < date.fromisoformat(event["effective_date"]):
                f["legacy_requirement_met"] = pattern_code(index, event, dom, "legacy_requirement_met", 3) != 1
            for required_fact in dom["secondary_gate"].split("+"):
                f[required_fact] = pattern_code(index, event, dom, f"transition:{required_fact}", 3) != 1
    else:
        raise ValueError(kind)
    return f


def active_new(event: dict[str, Any], facts: dict[str, Any], version: str) -> bool:
    return version == "new" and date.fromisoformat(facts["action_date"]) >= date.fromisoformat(event["effective_date"])


def state(decision: str, handling: str, obligations: dict[str, str], procedures: list[str],
          permissions: list[str], prohibitions: list[str], review: dict[str, Any],
          basis: list[str]) -> dict[str, Any]:
    return {"final_decision": decision, "current_handling": handling,
            "obligation_status": dict(sorted(obligations.items())),
            "obligations": sorted(obligations), "procedures": sorted(set(procedures)),
            "permissions": sorted(set(permissions)), "prohibitions": sorted(set(prohibitions)),
            "human_review": review, "basis": sorted(set(basis))}


def evaluate(event: dict[str, Any], dom: dict[str, Any], facts: dict[str, Any], version: str,
             exception_override: bool | None = None) -> dict[str, Any]:
    """Forward interpreter. Outputs only formal state, never a reference essay."""
    active = active_new(event, facts, version)
    old_req, new_req = {}, {},
    obligations = {"invariant:" + dom["invariant_obligation"]: "ongoing"}
    procedures: list[str] = []
    permissions: list[str] = []
    prohibitions: list[str] = []
    review = {"required": False, "reason": None, "status": "not_required"}
    basis = ["action_date", "upper_rule_version", "domain_rule"]
    kind, mode = event["kind"], dom["mode"]
    if kind == "age_threshold":
        age = age_at(facts["birth_date"], facts["action_date"])
        threshold = event["mechanism"]["new_threshold_years"] if active else event["mechanism"]["old_threshold_years"]
        exception = bool(facts.get("grandfathered_credential", False) and dom["has_downstream_exception"])
        if exception_override is not None:
            exception = exception_override
        qualified = age >= threshold or exception
        basis += ["birth_date", "action_date", "age_threshold"]
        if "secondary_gate_met" in facts: basis.append("secondary_gate_met")
        if exception: basis.append("grandfathered_credential")
        secondary = facts.get("secondary_gate_met", True)
        if mode == "permit":
            approved = qualified and secondary
            decision = dom["positive_decision"] if approved else dom["negative_decision"]
            handling = "start_independent_operation" if approved else "simulation_or_training_only"
            obligations["prerequisite:hardware_certificate"] = "fulfilled" if secondary else "pending"
            if not approved: obligations["route:simulator_training"] = "pending"; procedures.append("simulator_training")
        elif mode == "authority":
            if qualified:
                decision, handling = dom["positive_decision"], "self_authorized_export"
            else:
                decision = "joint_authority_confirmed" if facts["coapproval_done"] else dom["negative_decision"]
                handling = "release_with_joint_authority" if facts["coapproval_done"] else "hold_for_coapproval"
                obligations["procedure:coapproval"] = "fulfilled" if facts["coapproval_done"] else "pending"
                procedures.append("coapproval")
        elif mode == "role":
            qualified = qualified and facts["field_training_complete"]
            decision = dom["positive_decision"] if qualified else dom["negative_decision"]
            handling = "assign_field_lead" if qualified else "assign_support_role"
            obligations["prerequisite:field_training"] = "fulfilled" if facts["field_training_complete"] else "pending"
        elif mode == "dispatch":
            qualified = qualified and facts["secondary_gate_met"]
            if qualified: decision, handling = dom["positive_decision"], "execute_dispatch"
            elif facts["supervision_available"]: decision, handling = "shadow_dispatch_only", "supervised_shadow_seat"
            else: decision, handling = dom["negative_decision"], "hold_for_dispatch_supervisor"
            obligations["prerequisite:dispatch_exam"] = "fulfilled" if facts["secondary_gate_met"] else "pending"
            if not qualified: procedures.append("supervisor_assignment")
        elif mode == "lending":
            approved = facts["request_mode"] == "onsite" or qualified
            decision = dom["positive_decision"] if approved else dom["negative_decision"]
            handling = "onsite_reading" if facts["request_mode"] == "onsite" else ("release_for_offsite_loan" if approved else "deny_offsite_loan")
            if facts["request_mode"] == "offsite": obligations["procedure:chain_of_custody"] = "fulfilled" if approved else "pending"; procedures.append("chain_of_custody")
        elif mode == "remote":
            approved = qualified and facts["secondary_gate_met"]
            if approved: decision, handling = dom["positive_decision"], "remote_control"
            elif facts["supervision_available"]: decision, handling = "onsite_supervision_only", "local_supervised_control"
            else: decision, handling = dom["negative_decision"], "hold_remote_control"
            obligations["prerequisite:remote_operator_exam"] = "fulfilled" if facts["secondary_gate_met"] else "pending"
            if not approved: procedures.append("onsite_supervision")
        else: raise ValueError(mode)
        action = dom["action"]
        if handling in {"start_independent_operation", "self_authorized_export", "release_with_joint_authority", "assign_field_lead", "execute_dispatch", "release_for_offsite_loan", "remote_control"}:
            permissions.append(action)
        else: prohibitions.append(action)
    elif kind == "scope_expansion":
        old_features = set(event["mechanism"]["old_sensitive_features"])
        new_feature = event["mechanism"]["added_feature"]
        feature = facts["feature_type"]
        sensitive = feature in old_features or (active and feature == new_feature)
        in_scope = sensitive and facts["identity_inference"] and facts["identity_feature_generated"]
        control_values = {"separate_consent": facts.get("separate_consent", True),
                          "encrypted": facts.get("encrypted", True),
                          "purpose_match": facts.get("purpose_matches_domain", True),
                          "retention_valid": facts.get("retention_days", 0) <= 30}
        required_controls = dom["secondary_gate"].split("+") if mode != "biometric_transcript" else []
        controls = all(control_values[c] for c in required_controls) if in_scope else True
        basis += ["feature_type", "identity_inference", "identity_feature_generated"]
        if in_scope:
            basis += [k for k in ("separate_consent", "encrypted", "purpose_matches_domain", "retention_days") if k in facts]
        if in_scope:
            fact_key = {"separate_consent": "separate_consent", "encrypted": "encrypted",
                        "purpose_match": "purpose_matches_domain", "retention_valid": "retention_days"}
            for control in required_controls:
                key = fact_key[control]
                ok = control_values[control]
                obligation = "retention_within_30_days" if control == "retention_valid" else fact_key[control]
                obligations["control:" + obligation] = "fulfilled" if ok else "pending"
            procedures.extend(["feature_scope_review", "purpose_and_retention_check"])
            authorized = controls
        else:
            authorized = True
        if mode == "biometric_dispatch":
            decision, handling = (dom["positive_decision"], "execute_voice_dispatch") if authorized else (dom["negative_decision"], "manual_dispatch")
        elif mode == "biometric_entry":
            decision, handling = (dom["positive_decision"], "voice_entry") if authorized else (dom["negative_decision"], "badge_entry")
        elif mode == "biometric_profile":
            decision, handling = (dom["positive_decision"], "personalized_search") if authorized else (dom["negative_decision"], "generic_search")
        elif mode == "biometric_release":
            decision, handling = (dom["positive_decision"], "voice_release") if authorized else (dom["negative_decision"], "one_time_code_handover")
        elif mode == "biometric_records":
            decision, handling = (dom["positive_decision"], "self_service_record_access") if authorized else (dom["negative_decision"], "staff_verified_access")
        elif mode == "biometric_transcript":
            decision, handling = (dom["positive_decision"], "content_transcription_only") if authorized else (dom["negative_decision"], "identity_matching_disabled")
        else: raise ValueError(mode)
        if authorized: permissions.append(dom["action"])
        else: prohibitions.append(dom["action"]); obligations["procedure:" + dom["fallback_action"]] = "pending"; procedures.append(dom["fallback_action"])
    elif kind == "review_strengthening":
        impact_key = dom["impact_fact"] or "high_impact"
        triggered = facts[impact_key] and facts["adverse_proposed"]
        meaningful = triggered and all(facts[k] for k in ("materials_checked", "system_reason_checked", "reviewer_can_override", "review_reason_recorded"))
        standard = "meaningful" if active else "formal"
        completed = ((meaningful and facts["domain_file_access"]) if active else facts["formal_confirmation"]) if triggered else False
        basis += [impact_key, "adverse_proposed"]
        if triggered:
            basis += ["formal_confirmation", "materials_checked", "system_reason_checked", "reviewer_can_override", "review_reason_recorded", "domain_file_access"]
        decision = "proposed_adverse_action" if triggered else "no_adverse_action"
        if triggered:
            review = {"required": True, "reason": "high_impact_adverse_action", "standard": standard, "status": "satisfied" if completed else "pending"}
            obligations["procedure:" + standard + "_review"] = "fulfilled" if completed else "pending"
            procedures.append(standard + "_review")
            if active:
                for k in ("materials_checked", "system_reason_checked", "reviewer_can_override", "review_reason_recorded", "domain_file_access"):
                    obligations["review_element:" + k] = "fulfilled" if facts[k] else "pending"
                    procedures.append(k)
            if completed: handling = "execute_proposed_action"
            else: handling = "hold_for_human_review"
        else:
            handling = "continue_low_impact_or_non_adverse_path"
        if handling == "execute_proposed_action": permissions.append(dom["action"])
        else: prohibitions.append(dom["action"])
    elif kind == "validity_window_shortening":
        limit = event["mechanism"]["new_validity_days"] if active else event["mechanism"]["old_validity_days"]
        valid = facts["days_since_last_check"] <= limit
        secondary = facts["secondary_gate_met"]
        basis += ["days_since_last_check", "validity_window", "secondary_gate_met"]
        if mode == "validity_hold":
            ready = valid and secondary
            decision, handling = (dom["positive_decision"], "deploy") if ready else (dom["negative_decision"], "recheck_before_deployment")
        elif mode == "validity_maintenance":
            ready = valid and secondary
            decision, handling = (dom["positive_decision"], "start_machine") if ready else (dom["negative_decision"], "technician_reinspection")
        elif mode == "validity_reconsent":
            ready = valid and facts["secondary_gate_met"]
            decision, handling = (dom["positive_decision"], "reuse_data") if ready else (dom["negative_decision"], "pause_for_purpose_review")
        elif mode == "validity_quarantine":
            ready = valid and secondary
            decision, handling = (dom["positive_decision"], "release_batch") if ready else (dom["negative_decision"], "quarantine_batch")
        elif mode == "validity_downgrade":
            ready = valid and secondary
            decision, handling = (dom["positive_decision"], "remote_write") if ready else (dom["negative_decision"], "read_only")
        elif mode == "validity_route":
            ready = valid and secondary
            decision, handling = (dom["positive_decision"], "enter_route") if ready else (dom["negative_decision"], "reroute_or_recheck")
        else:
            raise ValueError(f"unsupported validity mode: {mode}")
        obligations["current_check"] = "fulfilled" if valid else "pending"
        obligations["secondary_gate"] = "fulfilled" if secondary else "pending"
        if not ready: procedures.append("recheck_or_domain_specific_fallback")
        if ready: permissions.append(dom["action"])
        else: prohibitions.append(dom["action"])
    elif kind == "evidence_threshold_increase":
        threshold = event["mechanism"]["new_min_independent_sources"] if active else event["mechanism"]["old_min_independent_sources"]
        enough_sources = facts["independent_source_count"] >= threshold
        domain_condition = facts[dom["secondary_gate"]]
        proposition_supported = enough_sources and facts["core_claim_true"] and domain_condition
        basis += ["independent_source_count", "independence_of_sources", "core_claim_true", "evidence_threshold", dom["secondary_gate"]]
        if mode == "evidence_provenance":
            decision, handling = (dom["positive_decision"], "certify_origin") if proposition_supported else (dom["negative_decision"], "request_independent_provenance")
        elif mode == "evidence_diagnosis":
            decision, handling = (dom["positive_decision"], "repair_from_diagnosis") if proposition_supported else (dom["negative_decision"], "observe_and_retest")
        elif mode == "evidence_eligibility":
            decision, handling = (dom["positive_decision"], "confirm_eligibility") if proposition_supported else (dom["negative_decision"], "hold_eligibility")
        elif mode == "evidence_confirmation":
            decision, handling = (dom["positive_decision"], "publish_alert") if proposition_supported else (dom["negative_decision"], "independent_retest")
        elif mode == "evidence_clearance":
            decision, handling = (dom["positive_decision"], "release_equipment") if proposition_supported else (dom["negative_decision"], "maintain_isolation")
        elif mode == "evidence_claim":
            decision, handling = (dom["positive_decision"], "allocate_resource") if proposition_supported else (dom["negative_decision"], "hold_final_allocation")
        else: raise ValueError(mode)
        obligations["evidence_threshold"] = "fulfilled" if enough_sources else "pending"
        obligations["domain_requirement:" + dom["secondary_gate"]] = "fulfilled" if domain_condition else "pending"
        if not enough_sources: procedures.append("obtain_independent_source")
        if proposition_supported: permissions.append(dom["action"])
        else: prohibitions.append(dom["action"])
    elif kind == "permission_default_reversal":
        if active:
            authorized = facts["affirmative_authorization"] and facts["authorization_scope_matches"] and not facts["explicit_objection"]
        else:
            authorized = not facts["explicit_objection"]
        basis += ["affirmative_authorization", "explicit_objection", "response_received", "authorization_scope_matches", "permission_default"]
        if mode == "permission_reuse":
            denied = dom["negative_decision"] if not authorized else dom["positive_decision"]
            decision, handling = (denied, "research_reuse") if authorized else (denied, "retain_service_only")
        elif mode == "permission_training": decision, handling = (dom["positive_decision"], "add_to_training_set") if authorized else (dom["negative_decision"], "exclude_from_training_set")
        elif mode == "permission_recontact": decision, handling = (dom["positive_decision"], "send_research_invitation") if authorized else (dom["negative_decision"], "do_not_contact")
        elif mode == "permission_analytics": decision, handling = (dom["positive_decision"], "behavioral_analysis") if authorized else (dom["negative_decision"], "safety_logs_only")
        elif mode == "permission_indexing": decision, handling = (dom["positive_decision"], "index_content") if authorized else (dom["negative_decision"], "keep_out_of_index")
        elif mode == "permission_sharing": decision, handling = (dom["positive_decision"], "share_across_projects") if authorized else (dom["negative_decision"], "keep_local")
        else: raise ValueError(mode)
        obligations["purpose_authorization"] = "fulfilled" if authorized else "pending"
        if authorized: permissions.append(dom["action"])
        else: prohibitions.append(dom["action"]); procedures.append(dom["fallback_action"])
    elif kind == "exception_narrowing":
        old_exception = facts["urgent"]
        new_exception = facts["urgent"] and facts["immediate_safety_risk"] and not facts["safe_alternative_available"]
        if exception_override is not None: old_exception = new_exception = exception_override
        exception = new_exception if active else old_exception
        restricted = facts["restricted_condition"]
        allowed = not restricted or exception
        basis += ["restricted_condition", "urgent", "immediate_safety_risk", "safe_alternative_available", "exception_scope"]
        if mode == "exception_control": decision, handling = (dom["positive_decision"], "remote_control") if allowed else (dom["negative_decision"], "dispatch_on_site_team")
        elif mode == "exception_energy": decision, handling = (dom["positive_decision"], "switch_mode") if allowed else (dom["negative_decision"], "retain_safe_mode")
        elif mode == "exception_shutdown": decision, handling = (dom["positive_decision"], "release_lock") if allowed else (dom["negative_decision"], "keep_shutdown_lock")
        elif mode == "exception_launch": decision, handling = (dom["positive_decision"], "launch") if allowed else (dom["negative_decision"], "ground_survey")
        elif mode == "exception_transfer": decision, handling = (dom["positive_decision"], "transfer_container") if allowed else (dom["negative_decision"], "seal_container")
        elif mode == "exception_route": decision, handling = (dom["positive_decision"], "cross_barrier") if allowed else (dom["negative_decision"], "reroute")
        else: raise ValueError(mode)
        if allowed: permissions.append(dom["action"])
        else: prohibitions.append(dom["action"]); procedures.append(dom["fallback_action"])
        if restricted: obligations["exception_basis_record"] = "fulfilled" if facts["qualified_responder_available"] else "pending"
    elif kind == "conflict_priority_change":
        conflict = facts["direct_conflict"] and facts["upper_safety_restriction_applies"] and facts["local_exception_allows_action"]
        if conflict:
            allowed = facts["local_exception_allows_action"] if not active else not facts["upper_safety_restriction_applies"]
        elif facts["upper_safety_restriction_applies"]:
            allowed = False
        else:
            allowed = facts["local_exception_allows_action"]
        if facts["independent_safeguard_complete"] and not facts["upper_safety_restriction_applies"]: allowed = True
        basis += ["direct_conflict", "upper_safety_restriction_applies", "local_exception_allows_action", "conflict_priority", "independent_safeguard_complete"]
        decision, handling = (dom["positive_decision"], "perform_local_action") if allowed else (dom["negative_decision"], "apply_upper_safety_route")
        if conflict: obligations["conflict_resolution_record"] = "fulfilled" if facts["independent_safeguard_complete"] else "pending"; procedures.append("record_conflict_resolution")
        if allowed: permissions.append(dom["action"])
        else: prohibitions.append(dom["action"]); procedures.append(dom["fallback_action"])
    elif kind == "transition_scope_expansion":
        effective = date.fromisoformat(event["effective_date"])
        filed_after = date.fromisoformat(facts["filed_date"]) >= effective
        pending = facts["request_status"] == "pending"
        executed = facts["request_status"] == "executed"
        if active:
            applies = filed_after or (pending and not executed)
        else:
            applies = filed_after
        if executed:
            requirement_met = True
            decision, handling = "already_executed_no_retroactive_reversal", "preserve_completed_action"
        else:
            requirement_met = all(facts[k] for k in dom["secondary_gate"].split("+")) if applies else facts["legacy_requirement_met"]
            if not applies:
                decision, handling = "legacy_standard_satisfied" if requirement_met else "legacy_standard_pending", "continue_under_legacy_standard" if requirement_met else "hold_for_legacy_requirement"
            else:
                decision, handling = (dom["positive_decision"], "execute_after_new_standard") if requirement_met else (dom["negative_decision"], "hold_for_new_standard")
        basis += ["filed_date", "request_status", "action_date", "transition_scope"]
        if applies and not executed: basis += dom["secondary_gate"].split("+")
        if not applies and not executed: basis.append("legacy_requirement_met")
        obligations["applicable_requirement"] = "fulfilled" if requirement_met else "pending"
        if applies and not requirement_met: procedures.append("complete_new_requirement")
        if handling in {"execute_after_new_standard", "continue_under_legacy_standard", "preserve_completed_action"}:
            permissions.append(dom["action"])
        else: prohibitions.append(dom["action"])
    else:
        raise ValueError(kind)
    return state(decision, handling, obligations, procedures, permissions, prohibitions, review, basis)


def oracle(event: dict[str, Any], dom: dict[str, Any], facts: dict[str, Any], version: str,
           exception_override: bool | None = None) -> dict[str, Any]:
    """Independent direct evaluator; does not call evaluate() or its helpers."""
    effective = date.fromisoformat(event["effective_date"])
    active = version == "new" and date.fromisoformat(facts["action_date"]) >= effective
    if event["kind"] == "age_threshold":
        b, a = date.fromisoformat(facts["birth_date"]), date.fromisoformat(facts["action_date"])
        age = a.year - b.year - ((a.month, a.day) < (b.month, b.day))
        need = 20 if active else 18
        grandfathered = bool(facts.get("grandfathered_credential", False) and dom["has_downstream_exception"])
        if exception_override is not None: grandfathered = exception_override
        eligible = age >= need or grandfathered
        if dom["mode"] == "permit":
            ok = eligible and facts["secondary_gate_met"]
            return {"decision": dom["positive_decision"] if ok else dom["negative_decision"], "handling": "start_independent_operation" if ok else "simulation_or_training_only"}
        if dom["mode"] == "authority":
            if eligible: return {"decision": dom["positive_decision"], "handling": "self_authorized_export"}
            return {"decision": "joint_authority_confirmed" if facts["coapproval_done"] else dom["negative_decision"], "handling": "release_with_joint_authority" if facts["coapproval_done"] else "hold_for_coapproval"}
        if dom["mode"] == "role":
            ok = eligible and facts["field_training_complete"]
            return {"decision": dom["positive_decision"] if ok else dom["negative_decision"], "handling": "assign_field_lead" if ok else "assign_support_role"}
        if dom["mode"] == "dispatch":
            ok = eligible and facts["secondary_gate_met"]
            if ok: return {"decision": dom["positive_decision"], "handling": "execute_dispatch"}
            if facts["supervision_available"]: return {"decision": "shadow_dispatch_only", "handling": "supervised_shadow_seat"}
            return {"decision": dom["negative_decision"], "handling": "hold_for_dispatch_supervisor"}
        if dom["mode"] == "lending":
            ok = facts["request_mode"] == "onsite" or eligible
            return {"decision": dom["positive_decision"] if ok else dom["negative_decision"], "handling": "onsite_reading" if facts["request_mode"] == "onsite" else ("release_for_offsite_loan" if ok else "deny_offsite_loan")}
        if dom["mode"] == "remote":
            ok = eligible and facts["secondary_gate_met"]
            if ok: return {"decision": dom["positive_decision"], "handling": "remote_control"}
            if facts["supervision_available"]: return {"decision": "onsite_supervision_only", "handling": "local_supervised_control"}
            return {"decision": dom["negative_decision"], "handling": "hold_remote_control"}
    elif event["kind"] == "scope_expansion":
        sensitive = facts["feature_type"] in {"face_template", "fingerprint_template"} or (active and facts["feature_type"] == "voiceprint")
        scope = sensitive and facts["identity_inference"] and facts["identity_feature_generated"]
        control_values = {"separate_consent": facts.get("separate_consent", True), "encrypted": facts.get("encrypted", True),
                          "purpose_match": facts.get("purpose_matches_domain", True), "retention_valid": facts.get("retention_days", 0) <= 30}
        required_controls = dom["secondary_gate"].split("+") if dom["mode"] != "biometric_transcript" else []
        ok = not scope or all(control_values[c] for c in required_controls)
        handling_by_mode = {"biometric_dispatch": ("execute_voice_dispatch", "manual_dispatch"), "biometric_entry": ("voice_entry", "badge_entry"), "biometric_profile": ("personalized_search", "generic_search"), "biometric_release": ("voice_release", "one_time_code_handover"), "biometric_records": ("self_service_record_access", "staff_verified_access"), "biometric_transcript": ("content_transcription_only", "identity_matching_disabled")}
        allow_h, deny_h = handling_by_mode[dom["mode"]]
        return {"decision": dom["positive_decision"] if ok else dom["negative_decision"], "handling": allow_h if ok else deny_h}
    elif event["kind"] == "review_strengthening":
        impact_key = dom["impact_fact"] or "high_impact"
        trigger = facts[impact_key] and facts["adverse_proposed"]
        meaningful = trigger and all(facts[k] for k in ("materials_checked", "system_reason_checked", "reviewer_can_override", "review_reason_recorded", "domain_file_access"))
        completed = (meaningful if active else facts["formal_confirmation"]) if trigger else False
        return {"decision": "proposed_adverse_action" if trigger else "no_adverse_action", "handling": ("execute_proposed_action" if completed else "hold_for_human_review") if trigger else "continue_low_impact_or_non_adverse_path"}
    elif event["kind"] == "validity_window_shortening":
        valid = facts["days_since_last_check"] <= (90 if active else 180)
        ok = valid and facts["secondary_gate_met"]
        mapping = {"validity_hold": ("deploy", "recheck_before_deployment"), "validity_maintenance": ("start_machine", "technician_reinspection"), "validity_reconsent": ("reuse_data", "pause_for_purpose_review"), "validity_quarantine": ("release_batch", "quarantine_batch"), "validity_downgrade": ("remote_write", "read_only"), "validity_route": ("enter_route", "reroute_or_recheck")}
        if dom["mode"] not in mapping:
            raise ValueError(f"unsupported validity mode: {dom['mode']}")
        on, off = mapping[dom["mode"]]
        return {"decision": dom["positive_decision"] if ok else dom["negative_decision"], "handling": on if ok else off}
    elif event["kind"] == "evidence_threshold_increase":
        count_ok = facts["independent_source_count"] >= (2 if active else 1)
        ok = count_ok and facts["core_claim_true"] and facts[dom["secondary_gate"]]
        mapping = {"evidence_provenance": ("certify_origin", "request_independent_provenance"), "evidence_diagnosis": ("repair_from_diagnosis", "observe_and_retest"), "evidence_eligibility": ("confirm_eligibility", "hold_eligibility"), "evidence_confirmation": ("publish_alert", "independent_retest"), "evidence_clearance": ("release_equipment", "maintain_isolation"), "evidence_claim": ("allocate_resource", "hold_final_allocation")}
        on, off = mapping[dom["mode"]]
        return {"decision": dom["positive_decision"] if ok else dom["negative_decision"], "handling": on if ok else off}
    elif event["kind"] == "permission_default_reversal":
        ok = ((facts["affirmative_authorization"] and facts["authorization_scope_matches"] and not facts["explicit_objection"]) if active else not facts["explicit_objection"])
        mapping = {"permission_reuse": ("research_reuse", "retain_service_only"), "permission_training": ("add_to_training_set", "exclude_from_training_set"), "permission_recontact": ("send_research_invitation", "do_not_contact"), "permission_analytics": ("behavioral_analysis", "safety_logs_only"), "permission_indexing": ("index_content", "keep_out_of_index"), "permission_sharing": ("share_across_projects", "keep_local")}
        on, off = mapping[dom["mode"]]
        return {"decision": dom["positive_decision"] if ok else dom["negative_decision"], "handling": on if ok else off}
    elif event["kind"] == "exception_narrowing":
        exception_old = facts["urgent"]
        exception_new = facts["urgent"] and facts["immediate_safety_risk"] and not facts["safe_alternative_available"]
        allowed = not facts["restricted_condition"] or (exception_new if active else exception_old)
        mapping = {"exception_control": ("remote_control", "dispatch_on_site_team"), "exception_energy": ("switch_mode", "retain_safe_mode"), "exception_shutdown": ("release_lock", "keep_shutdown_lock"), "exception_launch": ("launch", "ground_survey"), "exception_transfer": ("transfer_container", "seal_container"), "exception_route": ("cross_barrier", "reroute")}
        on, off = mapping[dom["mode"]]
        return {"decision": dom["positive_decision"] if allowed else dom["negative_decision"], "handling": on if allowed else off}
    elif event["kind"] == "conflict_priority_change":
        conflict = facts["direct_conflict"] and facts["upper_safety_restriction_applies"] and facts["local_exception_allows_action"]
        if conflict: allowed = facts["local_exception_allows_action"] if not active else not facts["upper_safety_restriction_applies"]
        elif facts["upper_safety_restriction_applies"]: allowed = False
        else: allowed = facts["local_exception_allows_action"]
        if facts["independent_safeguard_complete"] and not facts["upper_safety_restriction_applies"]: allowed = True
        return {"decision": dom["positive_decision"] if allowed else dom["negative_decision"], "handling": "perform_local_action" if allowed else "apply_upper_safety_route"}
    elif event["kind"] == "transition_scope_expansion":
        filed_after = date.fromisoformat(facts["filed_date"]) >= effective
        pending = facts["request_status"] == "pending"
        executed = facts["request_status"] == "executed"
        applies = (filed_after or (pending and not executed)) if active else filed_after
        if executed: return {"decision": "already_executed_no_retroactive_reversal", "handling": "preserve_completed_action"}
        met = all(facts[k] for k in dom["secondary_gate"].split("+")) if applies else facts["legacy_requirement_met"]
        if not applies: return {"decision": "legacy_standard_satisfied" if met else "legacy_standard_pending", "handling": "continue_under_legacy_standard" if met else "hold_for_legacy_requirement"}
        return {"decision": dom["positive_decision"] if met else dom["negative_decision"], "handling": "execute_after_new_standard" if met else "hold_for_new_standard"}
    else:
        raise ValueError(event["kind"])
    raise AssertionError("oracle has no result")


def oracle_full_state(event: dict[str, Any], dom: dict[str, Any], facts: dict[str, Any], version: str,
                      exception_override: bool | None = None) -> dict[str, Any]:
    """Independent reference projection for every non-decision state field.

    Final decision and handling are obtained from the separately implemented
    direct oracle above.  Obligations, procedures, permissions, review state,
    and reasoning basis are reconstructed here from facts and the rule profile,
    without calling the production evaluator or its state/delta helpers.
    """
    result = oracle(event, dom, facts, version, exception_override)
    decision, handling = result["decision"], result["handling"]
    effective = date.fromisoformat(event["effective_date"])
    active = version == "new" and date.fromisoformat(facts["action_date"]) >= effective
    kind, mode = event["kind"], dom["mode"]
    obs = {"invariant:" + dom["invariant_obligation"]: "ongoing"}
    procs: list[str] = []
    permissions: list[str] = []
    prohibitions: list[str] = []
    review = {"required": False, "reason": None, "status": "not_required"}
    basis = ["action_date", "upper_rule_version", "domain_rule"]

    if kind == "age_threshold":
        birth, action = date.fromisoformat(facts["birth_date"]), date.fromisoformat(facts["action_date"])
        age = action.year - birth.year - ((action.month, action.day) < (birth.month, birth.day))
        threshold = event["mechanism"]["new_threshold_years"] if active else event["mechanism"]["old_threshold_years"]
        grandfathered = bool(facts.get("grandfathered_credential", False) and dom["has_downstream_exception"])
        if exception_override is not None: grandfathered = exception_override
        qualified = age >= threshold or grandfathered
        basis += ["birth_date", "action_date", "age_threshold"]
        if "secondary_gate_met" in facts: basis.append("secondary_gate_met")
        if grandfathered: basis.append("grandfathered_credential")
        if mode == "permit":
            gate = facts.get("secondary_gate_met", True)
            obs["prerequisite:hardware_certificate"] = "fulfilled" if gate else "pending"
            if handling != "start_independent_operation": obs["route:simulator_training"] = "pending"; procs.append("simulator_training")
        elif mode == "authority" and not qualified:
            obs["procedure:coapproval"] = "fulfilled" if facts["coapproval_done"] else "pending"; procs.append("coapproval")
        elif mode == "role":
            obs["prerequisite:field_training"] = "fulfilled" if facts["field_training_complete"] else "pending"
        elif mode == "dispatch":
            obs["prerequisite:dispatch_exam"] = "fulfilled" if facts["secondary_gate_met"] else "pending"
            if handling != "execute_dispatch": procs.append("supervisor_assignment")
        elif mode == "lending" and facts["request_mode"] == "offsite":
            obs["procedure:chain_of_custody"] = "fulfilled" if handling == "release_for_offsite_loan" else "pending"; procs.append("chain_of_custody")
        elif mode == "remote":
            obs["prerequisite:remote_operator_exam"] = "fulfilled" if facts["secondary_gate_met"] else "pending"
            if handling != "remote_control": procs.append("onsite_supervision")
        allowed = handling in {"start_independent_operation", "self_authorized_export", "release_with_joint_authority", "assign_field_lead", "execute_dispatch", "release_for_offsite_loan", "remote_control"}
        (permissions if allowed else prohibitions).append(dom["action"])

    elif kind == "scope_expansion":
        old = set(event["mechanism"]["old_sensitive_features"])
        in_scope = (facts["feature_type"] in old or (active and facts["feature_type"] == event["mechanism"]["added_feature"])) and facts["identity_inference"] and facts["identity_feature_generated"]
        basis += ["feature_type", "identity_inference", "identity_feature_generated"]
        if in_scope:
            basis += [key for key in ("separate_consent", "encrypted", "purpose_matches_domain", "retention_days") if key in facts]
            control_to_fact = {"separate_consent": "separate_consent", "encrypted": "encrypted", "purpose_match": "purpose_matches_domain", "retention_valid": "retention_days"}
            for control in ([] if mode == "biometric_transcript" else dom["secondary_gate"].split("+")):
                fact_key = control_to_fact[control]
                ok = facts.get(fact_key, True)
                if control == "retention_valid": ok = facts.get("retention_days", 0) <= 30
                obligation = "retention_within_30_days" if control == "retention_valid" else fact_key
                obs["control:" + obligation] = "fulfilled" if ok else "pending"
            procs += ["feature_scope_review", "purpose_and_retention_check"]
        allowed = not prohibitions and decision == dom["positive_decision"]
        if allowed: permissions.append(dom["action"])
        else: prohibitions.append(dom["action"]); obs["procedure:" + dom["fallback_action"]] = "pending"; procs.append(dom["fallback_action"])

    elif kind == "review_strengthening":
        impact_key = dom["impact_fact"] or "high_impact"
        triggered = facts[impact_key] and facts["adverse_proposed"]
        basis += [impact_key, "adverse_proposed"]
        if triggered:
            basis += ["formal_confirmation", "materials_checked", "system_reason_checked", "reviewer_can_override", "review_reason_recorded", "domain_file_access"]
            standard = "meaningful" if active else "formal"
            complete = (all(facts[k] for k in ("materials_checked", "system_reason_checked", "reviewer_can_override", "review_reason_recorded", "domain_file_access")) if active else facts["formal_confirmation"])
            review = {"required": True, "reason": "high_impact_adverse_action", "standard": standard, "status": "satisfied" if complete else "pending"}
            obs["procedure:" + standard + "_review"] = "fulfilled" if complete else "pending"; procs.append(standard + "_review")
            if active:
                for key in ("materials_checked", "system_reason_checked", "reviewer_can_override", "review_reason_recorded", "domain_file_access"):
                    obs["review_element:" + key] = "fulfilled" if facts[key] else "pending"; procs.append(key)
        if handling == "execute_proposed_action": permissions.append(dom["action"])
        else: prohibitions.append(dom["action"])

    elif kind == "validity_window_shortening":
        valid = facts["days_since_last_check"] <= (90 if active else 180)
        basis += ["days_since_last_check", "validity_window", "secondary_gate_met"]
        obs["current_check"] = "fulfilled" if valid else "pending"
        obs["secondary_gate"] = "fulfilled" if facts["secondary_gate_met"] else "pending"
        if not valid or not facts["secondary_gate_met"]: procs.append("recheck_or_domain_specific_fallback")
        if handling in {"deploy", "start_machine", "reuse_data", "release_batch", "remote_write", "enter_route", "complete_domain_action"}: permissions.append(dom["action"])
        else: prohibitions.append(dom["action"])

    elif kind == "evidence_threshold_increase":
        enough = facts["independent_source_count"] >= (2 if active else 1)
        basis += ["independent_source_count", "independence_of_sources", "core_claim_true", "evidence_threshold", dom["secondary_gate"]]
        obs["evidence_threshold"] = "fulfilled" if enough else "pending"
        gate_ok = facts[dom["secondary_gate"]]
        obs["domain_requirement:" + dom["secondary_gate"]] = "fulfilled" if gate_ok else "pending"
        if not enough: procs.append("obtain_independent_source")
        if handling in {"certify_origin", "repair_from_diagnosis", "confirm_eligibility", "publish_alert", "release_equipment", "allocate_resource"}: permissions.append(dom["action"])
        else: prohibitions.append(dom["action"])

    elif kind == "permission_default_reversal":
        basis += ["affirmative_authorization", "explicit_objection", "response_received", "authorization_scope_matches", "permission_default"]
        allowed = (facts["affirmative_authorization"] and facts["authorization_scope_matches"] and not facts["explicit_objection"]) if active else not facts["explicit_objection"]
        obs["purpose_authorization"] = "fulfilled" if allowed else "pending"
        if allowed: permissions.append(dom["action"])
        else: prohibitions.append(dom["action"]); procs.append(dom["fallback_action"])

    elif kind == "exception_narrowing":
        basis += ["restricted_condition", "urgent", "immediate_safety_risk", "safe_alternative_available", "exception_scope"]
        if facts["restricted_condition"]:
            obs["exception_basis_record"] = "fulfilled" if facts["qualified_responder_available"] else "pending"
        if decision == dom["positive_decision"]: permissions.append(dom["action"])
        else: prohibitions.append(dom["action"]); procs.append(dom["fallback_action"])

    elif kind == "conflict_priority_change":
        conflict = facts["direct_conflict"] and facts["upper_safety_restriction_applies"] and facts["local_exception_allows_action"]
        basis += ["direct_conflict", "upper_safety_restriction_applies", "local_exception_allows_action", "conflict_priority", "independent_safeguard_complete"]
        if conflict: obs["conflict_resolution_record"] = "fulfilled" if facts["independent_safeguard_complete"] else "pending"; procs.append("record_conflict_resolution")
        if decision == dom["positive_decision"]: permissions.append(dom["action"])
        else: prohibitions.append(dom["action"]); procs.append(dom["fallback_action"])

    elif kind == "transition_scope_expansion":
        filed_after = date.fromisoformat(facts["filed_date"]) >= effective
        pending = facts["request_status"] == "pending"
        executed = facts["request_status"] == "executed"
        applies = (filed_after or (pending and not executed)) if active else filed_after
        basis += ["filed_date", "request_status", "action_date", "transition_scope"]
        if applies and not executed: basis += dom["secondary_gate"].split("+")
        if not applies and not executed: basis.append("legacy_requirement_met")
        met = True if executed else (all(facts[k] for k in dom["secondary_gate"].split("+")) if applies else facts["legacy_requirement_met"])
        obs["applicable_requirement"] = "fulfilled" if met else "pending"
        if applies and not met: procs.append("complete_new_requirement")
        if handling in {"execute_after_new_standard", "continue_under_legacy_standard", "preserve_completed_action"}: permissions.append(dom["action"])
        else: prohibitions.append(dom["action"])
    else:
        raise ValueError(kind)

    return {"final_decision": decision, "current_handling": handling,
            "obligation_status": dict(sorted(obs.items())), "obligations": sorted(obs),
            "procedures": sorted(set(procs)), "permissions": sorted(set(permissions)),
            "prohibitions": sorted(set(prohibitions)), "human_review": review,
            "basis": sorted(set(basis))}


def make_delta(old: dict[str, Any], new: dict[str, Any], event: dict[str, Any], dom: dict[str, Any], facts: dict[str, Any]) -> dict[str, Any]:
    old_ob, new_ob = set(old["obligations"]), set(new["obligations"])
    old_pr, new_pr = set(old["procedures"]), set(new["procedures"])
    changed = old["final_decision"] != new["final_decision"]
    operational = any((old["current_handling"] != new["current_handling"], old_ob != new_ob,
                       old_pr != new_pr, old["permissions"] != new["permissions"],
                       old["prohibitions"] != new["prohibitions"], old["human_review"] != new["human_review"],
                       old["obligation_status"] != new["obligation_status"]))
    exception_preserves = False
    if (event["kind"] == "age_threshold" and dom["has_downstream_exception"]
            and facts.get("grandfathered_credential", False) and active_new(event, facts, "new")):
        without_new = evaluate(event, dom, facts, "new", exception_override=False)
        without_old = evaluate(event, dom, facts, "old", exception_override=False)
        exception_preserves = (old == new and without_new != without_old)
    result_type = "decision_changed" if changed else "procedure_only" if operational else "exception_preserves" if exception_preserves else "unaffected"
    return {
        "final_decision_changed": changed,
        "current_handling_changed": old["current_handling"] != new["current_handling"],
        "obligations_added": sorted(new_ob - old_ob),
        "obligations_removed": sorted(old_ob - new_ob),
        "obligations_unchanged": sorted(old_ob & new_ob),
        "obligation_status_changed": {k: {"old": old["obligation_status"][k], "new": new["obligation_status"][k]}
                                       for k in sorted(old_ob & new_ob) if old["obligation_status"][k] != new["obligation_status"][k]},
        "procedures_added": sorted(new_pr - old_pr), "procedures_removed": sorted(old_pr - new_pr),
        "procedures_unchanged": sorted(old_pr & new_pr),
        "permissions_added": sorted(set(new["permissions"]) - set(old["permissions"])),
        "permissions_removed": sorted(set(old["permissions"]) - set(new["permissions"])),
        "prohibitions_added": sorted(set(new["prohibitions"]) - set(old["prohibitions"])),
        "prohibitions_removed": sorted(set(old["prohibitions"]) - set(new["prohibitions"])),
        "human_review_changed": old["human_review"] != new["human_review"],
        "exception_blocks_effect": exception_preserves,
        "result_type": result_type,
    }


def oracle_delta(old: dict[str, Any], new: dict[str, Any], event: dict[str, Any],
                 dom: dict[str, Any], facts: dict[str, Any]) -> dict[str, Any]:
    """Reference set-difference implementation, independent of make_delta()."""
    old_ob, new_ob = set(old["obligations"]), set(new["obligations"])
    old_pr, new_pr = set(old["procedures"]), set(new["procedures"])
    old_permissions, new_permissions = set(old["permissions"]), set(new["permissions"])
    old_prohibitions, new_prohibitions = set(old["prohibitions"]), set(new["prohibitions"])
    decision_changed = old["final_decision"] != new["final_decision"]
    status_changed = {key: {"old": old["obligation_status"][key], "new": new["obligation_status"][key]}
                      for key in sorted(old_ob & new_ob)
                      if old["obligation_status"][key] != new["obligation_status"][key]}
    any_other_change = any((old["current_handling"] != new["current_handling"], old_ob != new_ob,
                            old_pr != new_pr, old_permissions != new_permissions,
                            old_prohibitions != new_prohibitions,
                            old["human_review"] != new["human_review"], bool(status_changed)))
    exception_blocks = False
    if (event["kind"] == "age_threshold" and dom["has_downstream_exception"]
            and facts.get("grandfathered_credential", False)
            and date.fromisoformat(facts["action_date"]) >= date.fromisoformat(event["effective_date"])):
        without_exception_old = oracle_full_state(event, dom, facts, "old", exception_override=False)
        without_exception_new = oracle_full_state(event, dom, facts, "new", exception_override=False)
        exception_blocks = old == new and without_exception_old != without_exception_new
    result_type = ("decision_changed" if decision_changed else "procedure_only" if any_other_change
                   else "exception_preserves" if exception_blocks else "unaffected")
    return {
        "final_decision_changed": decision_changed,
        "current_handling_changed": old["current_handling"] != new["current_handling"],
        "obligations_added": sorted(new_ob - old_ob),
        "obligations_removed": sorted(old_ob - new_ob),
        "obligations_unchanged": sorted(old_ob & new_ob),
        "obligation_status_changed": status_changed,
        "procedures_added": sorted(new_pr - old_pr),
        "procedures_removed": sorted(old_pr - new_pr),
        "procedures_unchanged": sorted(old_pr & new_pr),
        "permissions_added": sorted(new_permissions - old_permissions),
        "permissions_removed": sorted(old_permissions - new_permissions),
        "prohibitions_added": sorted(new_prohibitions - old_prohibitions),
        "prohibitions_removed": sorted(old_prohibitions - new_prohibitions),
        "human_review_changed": old["human_review"] != new["human_review"],
        "exception_blocks_effect": exception_blocks,
        "result_type": result_type,
    }


def domain_rule_text(event: dict[str, Any], dom: dict[str, Any]) -> str:
    kind, mode = event["kind"], dom["mode"]
    if kind == "age_threshold":
        if mode == "permit": rule = "按操作日适用的上层年龄标准判断独立操作资格；达到该标准且设备安全证书有效时，允许独立操作，否则只能使用模拟器。"
        elif mode == "authority": rule = "达到操作日适用的上层年龄标准者可自行授权数据导出；未达到者须完成共同授权，否则不得导出。"
        elif mode == "role": rule = "通常须达到操作日适用的上层年龄标准并完成野外训练，方可担任领队；否则安排为协作员。"
        elif mode == "dispatch": rule = "达到操作日适用的上层年龄标准并通过调度考试者可直接调度；否则仅可在监督席位影子操作，且无监督席位时暂停。"
        elif mode == "lending": rule = "馆内阅览不受年龄资格限制；带离阅览室的外借须达到操作日适用的上层年龄标准，否则仅可馆内阅览。"
        elif mode == "remote": rule = "达到操作日适用的上层年龄标准且通过远程操控考试者可远程控制；否则只可现场监督操作，未安排监督时不得操作。"
        else: raise ValueError(mode)
    elif kind == "scope_expansion":
        if mode == "biometric_transcript":
            rule = "仅转写内容且不推断身份、不生成身份特征时，按内容处理，不触发本次身份特征控制；不得据此建立说话人身份匹配。"
        else:
            control_names = {"separate_consent": "单独同意", "encrypted": "加密存储",
                             "purpose_match": "用途与登记一致", "retention_valid": "保存不超过30日"}
            required = [control_names[x] for x in dom["secondary_gate"].split("+")]
            rule = ("身份推断且生成身份特征时，若该特征属于当前上层规则覆盖的敏感生物识别信息，必须同时满足本领域控制条件："
                    + "、".join(required) + "；任一项不满足时采用本领域替代路径。")
            rule += "仅转写内容且不推断身份、不生成身份特征时，不触发本次身份特征控制。"
    elif kind == "review_strengthening":
        rule = ("只有高影响且提出不利处理时才触发复核；不满足这两个条件时按低影响或非不利路径继续。触发后，按操作日适用的上层规则完成相应复核标准；未完成前不得执行不利处理。"
                "本领域要求审查人能够访问个案材料。")
    elif kind == "validity_window_shortening":
        rule = "核验距操作日未超过上层规则规定的有效期，且本领域附加条件满足，方可执行该操作；否则暂停并走本领域重新核验或降级路径。"
    elif kind == "evidence_threshold_increase":
        rule = "关键事实为真且得到上层规则要求数量的独立、可核验证据支持时，方可确认并执行；否则保持待核状态并补充独立证据。"
    elif kind == "permission_default_reversal":
        rule = "仅当上层规则要求的授权条件满足、授权范围与本次用途一致且没有明确拒绝时，才可开展本领域的二次用途；否则停止二次用途并按替代路径处理。"
    elif kind == "exception_narrowing":
        rule = "受限制操作通常不得实施；只有符合操作日适用的上层紧急例外条件时才可继续，否则执行本领域安全替代路径。"
    elif kind == "conflict_priority_change":
        rule = "发生直接冲突时，按操作日适用的上层冲突优先级解决；不存在直接冲突时分别遵守各自适用规则。独立防护完成不授权违反仍然适用的上层安全限制。"
    elif kind == "transition_scope_expansion":
        rule = "先根据申请日期、待处理/已执行状态和操作日确定是否适用新标准；适用时须满足新要求后才能办理，已执行事项不追溯撤销；不适用时按旧标准处理。"
    else: raise ValueError(kind)
    return (f"本领域：{dom['name']}。决策链：{dom['decision_chain']}。{rule}"
            f"操作对象：{dom['action']}；不满足条件时的处置：{dom['fallback_action']}。"
            f"始终保留的事项：{dom['invariant_obligation']}。"
            + (f"本领域另有规则：{dom['local_rule']}" if dom["local_rule"] else ""))


def clue_lines(event: dict[str, Any], dom: dict[str, Any], facts: dict[str, Any], difficulty: str) -> list[tuple[str, list[str]]]:
    kind = event["kind"]
    def yes(v: bool) -> str: return "已记录" if v else "未记录"
    if kind == "age_threshold":
        age = age_at(facts["birth_date"], facts["action_date"])
        fields = [("birth_date", f"出生登记日为{facts['birth_date']}"),
                  ("action_date", f"拟实施日为{facts['action_date']}")]
        if "secondary_gate_met" in facts: fields.append(("secondary_gate_met", f"领域资格台账{('显示前置资质齐备' if facts['secondary_gate_met'] else '显示前置资质尚缺')}"))
        if "supervision_available" in facts: fields.append(("supervision_available", f"值班表{('安排了现场监督员' if facts['supervision_available'] else '未安排现场监督员')}"))
        if "coapproval_done" in facts: fields.append(("coapproval_done", f"共同授权档案{('已归档' if facts['coapproval_done'] else '未归档')}"))
        if "request_mode" in facts: fields.append(("request_mode", "请求方式为馆内阅览" if facts["request_mode"] == "onsite" else "请求方式为外借"))
        if "field_training_complete" in facts: fields.append(("field_training_complete", f"野外训练记录{('齐全' if facts['field_training_complete'] else '未完成')}"))
        if "grandfathered_credential" in facts:
            credential = dom.get("grandfathered_credential_name") or ("领队证" if dom["mode"] == "role" else "特藏外借许可")
            fields.append(("grandfathered_credential", f"新标准生效前签发且当前仍有效的{credential}{('存在' if facts['grandfathered_credential'] else '不存在')}"))
        rendered = []
        for key, statement in fields:
            if key == "birth_date" and difficulty == "easy": statement += f"，按拟实施日计算已满{age}周岁"
            rendered.append((statement + "。", [key]))
        if difficulty == "easy": return [(" ".join(x[0] for x in rendered), [k for _, ks in rendered for k in ks])]
        if difficulty == "medium":
            split = max(2, len(rendered) // 2)
            return [("登记与计划记录：" + "；".join(x[0] for x in rendered[:split]), [k for _, ks in rendered[:split] for k in ks]),
                    ("领域台账：" + "；".join(x[0] for x in rendered[split:]), [k for _, ks in rendered[split:] for k in ks])]
        return [(f"人员档案：{rendered[0][0]}", rendered[0][1]),
                (f"操作计划：{rendered[1][0]}", rendered[1][1]),
                *[(f"关联记录：{t}", ks) for t, ks in (("；".join(x[0] for x in rendered[2:]), [k for _, ks in rendered[2:] for k in ks]),) if t]]
    if kind == "scope_expansion":
        labels = {"feature_type": "特征类型", "identity_inference": "身份推断", "identity_feature_generated": "身份特征生成", "separate_consent": "单独同意", "encrypted": "加密", "purpose_matches_domain": "用途与本领域匹配", "retention_days": "保存天数", "action_date": "操作日期"}
        def statement(k: str) -> str:
            v = facts[k]
            if difficulty == "easy":
                if k == "action_date": return f"操作日期为{v}"
                if k == "feature_type": return f"系统处理的特征类型为{'声纹' if v == 'voiceprint' else '普通语音转写'}"
                if k == "retention_days": return f"计划保存{v}日"
                return f"{labels[k]}{'已记录/满足' if v else '未记录/不满足'}"
            if k == "action_date": return f"处理工单日期栏记为{v}"
            if k == "feature_type": return f"输入流程标记为{v}"
            if k == "identity_inference": return "运行记录显示系统会把信号用于辨认具体说话者" if v else "运行记录显示系统不区分说话者身份"
            if k == "identity_feature_generated": return "输出档案保存了可重复识别个人的特征模板" if v else "输出档案没有可重复识别个人的特征模板"
            if k == "separate_consent": return "授权档案中找到本项用途单独授权" if v else "授权档案中找不到本项用途的单独授权"
            if k == "encrypted": return "存储审计记录显示特征已加密" if v else "存储审计记录显示特征仍为未加密状态"
            if k == "purpose_matches_domain": return "登记用途与本次处理用途相同" if v else "登记用途不覆盖本次处理用途"
            if k == "retention_days": return f"删除计划排在处理后第{v}日"
            raise KeyError(k)
        keys = list(facts)
        if difficulty == "easy": return [("；".join(statement(k) for k in keys), keys)]
        if difficulty == "medium":
            pivot = min(4, max(2, len(keys) // 2))
            return [("处理日志：" + "；".join(statement(k) for k in keys[:pivot]), keys[:pivot]),
                    ("授权与存储台账：" + "；".join(statement(k) for k in keys[pivot:]), keys[pivot:])]
        chunks = [keys[::3], keys[1::3], keys[2::3]]
        return [(f"来源记录{n+1}：" + "；".join(statement(k) for k in group), group) for n, group in enumerate(chunks) if group]
    if kind == "review_strengthening":
        impact_key = dom["impact_fact"] or "high_impact"
        keys = ["action_date", impact_key, "adverse_proposed"]
        if facts[impact_key] and facts["adverse_proposed"]:
            keys += ["formal_confirmation", "materials_checked", "system_reason_checked",
                     "reviewer_can_override", "review_reason_recorded", "domain_file_access"]
        def review_phrase(k: str) -> str:
            v = facts[k]
            if k == impact_key:
                impact_text = dom["impact_true_text"] if v else dom["impact_false_text"]
                prefix = {"easy": "本领域相关记录：", "medium": "业务资料核对显示：", "hard": "关联台账中可查到："}[difficulty]
                return prefix + impact_text
            if difficulty == "easy":
                if k == "action_date": return f"操作日期为{v}"
                if k == "adverse_proposed": return "系统提出不利处理" if v else "系统未提出不利处理"
                text = {"formal_confirmation": "仅点击确认", "materials_checked": "查看个案材料",
                        "system_reason_checked": "查看系统理由", "reviewer_can_override": "审查人可修改或推翻建议",
                        "review_reason_recorded": "记录审查理由", "domain_file_access": "可访问个案档案"}[k]
                return f"{text}{'：是' if v else '：否'}"
            if k == "action_date": return f"案件时间戳为{v}"
            if k == "adverse_proposed": return "建议结果将减少或撤销现有权益" if v else "建议结果没有减少或撤销权益"
            indirect = {"formal_confirmation": ("审计日志记录到确认按钮点击", "审计日志没有确认按钮点击"),
                        "materials_checked": ("个案档案访问日志显示材料页面被打开", "个案档案访问日志没有材料页面记录"),
                        "system_reason_checked": ("解释面板留下了查看记录", "解释面板没有查看记录"),
                        "reviewer_can_override": ("审查账户权限允许改写系统建议", "审查账户为只读权限"),
                        "review_reason_recorded": ("审查记录包含具体理由条目", "审查记录没有理由条目"),
                        "domain_file_access": ("审查人账号具有个案档案访问权限", "审查人账号没有个案档案访问权限")}
            return indirect[k][0 if v else 1]
        if difficulty == "easy":
            return [("；".join(review_phrase(k) for k in keys), keys)]
        if difficulty == "medium":
            pivot = 3
            return [("业务记录：" + "；".join(review_phrase(k) for k in keys[:pivot]), keys[:pivot]),
                    ("审查系统台账：" + "；".join(review_phrase(k) for k in keys[pivot:]), keys[pivot:])]
        chunks = [keys[::3], keys[1::3], keys[2::3]]
        return [(f"关联记录{n+1}：" + "；".join(review_phrase(k) for k in group), group) for n, group in enumerate(chunks) if group]
    # Other event families share a fact-to-text renderer but maintain distinct
    # controlled-language styles and evidence locations for each difficulty.
    labels = {"days_since_last_check":"距上次核验天数", "secondary_gate_met":"领域附加条件", "source_chain_intact":"材料来源链完整", "fault_reproduced":"故障在独立复测中复现", "budget_available":"本轮预算余额充足", "sampling_chain_intact":"采样封签与交接链完整", "post_repair_test_passed":"维修后独立负载测试通过", "claimant_identity_matches":"申领记录与申请人身份匹配", "independent_source_count":"独立证据来源数", "core_claim_true":"关键事实被支持", "authorization_record_exists":"存在授权记录", "affirmative_authorization":"明确肯定授权", "response_received":"是否收到回复", "explicit_objection":"明确拒绝", "authorization_scope_matches":"授权范围匹配", "restricted_condition":"限制条件存在", "urgent":"紧迫情形存在", "immediate_safety_risk":"即时安全风险存在", "safe_alternative_available":"安全替代方案可用", "qualified_responder_available":"合格响应人员可用", "upper_safety_restriction_applies":"上层安全限制适用", "local_exception_allows_action":"本地例外允许操作", "direct_conflict":"两条规则直接冲突", "independent_safeguard_complete":"独立防护已完成", "filed_date":"提交日期", "request_status":"申请状态", "legacy_requirement_met":"旧标准已满足", "safety_training_current":"安全培训记录在有效期内", "qualified_lead_assigned":"具资质的现场负责人已安排", "purpose_statement_present":"具体研究目的已说明", "field_scope_defined":"访问字段范围已列明", "deletion_plan_present":"删除计划已提供", "independent_budget_source_provided":"独立预算来源已证明", "disbursement_plan_signed":"拨付计划已签署", "storage_environment_committed":"保管环境承诺已提供", "handover_contact_named":"交接联系人已具名", "risk_notice_signed":"路线风险告知已签署", "return_contact_plan_present":"返程联络安排已提供", "independent_tech_review_complete":"独立技师复核已完成", "second_signoff_present":"第二名技师已签字", "action_date":"操作日期", "birth_date":"出生日期", "evidence_sources":"证据清单"}
    keys = list(facts)
    gate_keys = dom["secondary_gate"].split("+") if dom["secondary_gate"] else []
    gate_descriptions = dom.get("secondary_gate_label", "").split("+") if dom.get("secondary_gate_label") else []
    gate_label_by_key = dict(zip(gate_keys, gate_descriptions)) if len(gate_keys) == len(gate_descriptions) else {}
    def phrase(k: str) -> str:
        v = facts[k]
        if k in gate_label_by_key and isinstance(v, bool):
            return f"{gate_label_by_key[k]}{'已满足' if v else '尚未满足'}"
        if difficulty != "easy":
            if k == "days_since_last_check": return f"最近一次核验记录距本次操作约{v}日"
            if k == "independent_source_count": return f"可分别追溯的证据来源共{v}项"
            if k == "evidence_sources": return f"来源台账列有：{('、'.join(v) if v else '没有来源记录')}"
            if k == "core_claim_true": return "核验材料支持待确认的关键事实" if v else "核验材料没有支持待确认的关键事实"
            if k == "affirmative_authorization": return "授权档案中有本次用途的肯定授权" if v else "授权档案中没有本次用途的肯定授权"
            if k == "explicit_objection": return "记录中出现当事人的明确拒绝" if v else "没有找到当事人的明确拒绝记录"
            if k == "authorization_scope_matches": return "授权文件覆盖当前所述用途" if v else "授权文件指向另一用途，未覆盖当前处理"
            if k == "restricted_condition": return "操作对象列在受限清单内" if v else "操作对象不在受限清单内"
            if k == "urgent": return "值班记录表明不能按常规时限等待" if v else "记录未显示必须立即处置的时间压力"
            if k == "immediate_safety_risk": return "风险评估指出眼前存在人身安全危险" if v else "风险评估未指出眼前的人身安全危险"
            if k == "safe_alternative_available": return "现场仍有经验证的安全替代方案" if v else "现场没有可用的安全替代方案"
            if k == "qualified_responder_available": return "合格响应人员已在现场待命" if v else "现场没有合格响应人员待命"
            if k == "upper_safety_restriction_applies": return "上层安全清单覆盖本次操作" if v else "上层安全清单没有覆盖本次操作"
            if k == "local_exception_allows_action": return "本地操作手册为该步骤保留了例外通道" if v else "本地操作手册没有为该步骤保留例外"
            if k == "direct_conflict": return "两份适用指令对同一步骤给出相反要求" if v else "两份指令在本案步骤上没有相反要求"
            if k == "independent_safeguard_complete": return "附加防护清单的记录项目均已签核" if v else "附加防护清单仍有未签核项目"
            if k == "filed_date": return f"收件记录上的日期为{v}"
            if k == "request_status": return "队列记录显示该事项仍待办理" if v == "pending" else "结案日志显示该事项已完成执行"
            if k == "legacy_requirement_met": return "旧版核对表没有未完成项目" if v else "旧版核对表仍有未完成项目"
            if k == "action_date": return f"办理记录的操作日期为{v}"
            if k == "birth_date": return f"人员登记日为{v}"
            transition_phrases = {
                "safety_training_current": ("培训系统中的证书有效期覆盖办理日", "培训系统中的证书在办理日前已过期"),
                "qualified_lead_assigned": ("排班表列有具资质的现场负责人", "排班表没有具资质的现场负责人"),
                "purpose_statement_present": ("申请表说明了本次研究的具体目的", "申请表没有说明本次研究的具体目的"),
                "field_scope_defined": ("附件逐项列出可访问的数据字段", "附件没有界定可访问的数据字段"),
                "deletion_plan_present": ("材料中写明了数据删除时间表", "材料中没有数据删除时间表"),
                "independent_budget_source_provided": ("预算凭据来自独立财务来源", "没有独立财务来源的预算凭据"),
                "disbursement_plan_signed": ("拨付计划末页有负责人签署", "拨付计划末页没有负责人签署"),
                "storage_environment_committed": ("接收方的保管环境承诺已归档", "接收方没有提交保管环境承诺"),
                "handover_contact_named": ("交接表列出了具名联系人", "交接表没有具名联系人"),
                "risk_notice_signed": ("风险告知页留有申请人签名", "风险告知页未见申请人签名"),
                "return_contact_plan_present": ("返程联络表列有可用安排", "返程联络表没有可用安排"),
                "independent_tech_review_complete": ("复核日志显示由维修执行人以外的技师完成检查", "没有维修执行人以外的技师复核记录"),
                "second_signoff_present": ("设备放行单上有第二名技师签字", "设备放行单缺少第二名技师签字"),
            }
            if k in transition_phrases: return transition_phrases[k][0 if v else 1]
        if isinstance(v, bool):
            return f"{labels.get(k,k)}：{'是' if v else '否'}"
        if v is None: return f"{labels.get(k,k)}没有执行日期记录"
        if isinstance(v, list): return f"{labels.get(k,k)}：{('、'.join(v) if v else '没有来源记录')}"
        if k == "request_status": return "申请状态为待处理" if v == "pending" else "申请状态为已执行"
        return f"{labels.get(k,k)}为{v}"
    if difficulty == "easy":
        return [("；".join(phrase(k) for k in keys), keys)]
    if difficulty == "medium":
        mid = (len(keys) + 1) // 2
        return [("办理记录：" + "；".join(phrase(k) for k in keys[:mid]), keys[:mid]),
                ("关联台账：" + "；".join(phrase(k) for k in keys[mid:]), keys[mid:])]
    # Hard only changes the distribution across source records; it adds no facts.
    chunks = [keys[::3], keys[1::3], keys[2::3]]
    return [(f"来源记录{n+1}：" + "；".join(phrase(k) for k in group), group)
            for n, group in enumerate(chunks) if group]


def render_case(event: dict[str, Any], dom: dict[str, Any], facts: dict[str, Any], difficulty: str) -> tuple[str, list[dict[str, Any]]]:
    lines = clue_lines(event, dom, facts, difficulty)
    entries, chunks = [], []
    offset = 0
    for i, (text, keys) in enumerate(lines, 1):
        sentence = f"[F{i:03d}] {text}"
        chunks.append(sentence)
        entries.append({"evidence_id": f"F{i:03d}", "fact_keys": keys,
                        "text": text, "char_start": offset + len(f"[F{i:03d}] "),
                        "char_end": offset + len(sentence)})
        offset += len(sentence) + 1
    return "\n".join(chunks), entries


def result_type_for(event: dict[str, Any], dom: dict[str, Any], facts: dict[str, Any], old: dict[str, Any], new: dict[str, Any]) -> str:
    return make_delta(old, new, event, dom, facts)["result_type"]


def case_gold(family_id: str, event: dict[str, Any], dom: dict[str, Any], facts: dict[str, Any],
              old: dict[str, Any], new: dict[str, Any], delta: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    required = {
        "final_decision": new["final_decision"],
        "current_handling": new["current_handling"],
        "obligations_added": delta["obligations_added"],
        "obligations_removed": delta["obligations_removed"],
        "obligations_unchanged": delta["obligations_unchanged"],
        "obligation_status_changed": delta["obligation_status_changed"],
        "procedures_added": delta["procedures_added"],
        "procedures_removed": delta["procedures_removed"],
        "human_review": new["human_review"],
        "permissions": new["permissions"],
        "prohibitions": new["prohibitions"],
    }
    forbidden = []
    if new["prohibitions"]: forbidden.append({"claim": "assert_current_permission_for_prohibited_action", "actions": new["prohibitions"]})
    if new["permissions"]: forbidden.append({"claim": "assert_current_prohibition_for_permitted_action", "actions": new["permissions"]})
    if delta["obligations_removed"]: forbidden.append({"claim": "retain_cancelled_obligation_as_current", "obligations": delta["obligations_removed"]})
    if delta["obligations_added"]: forbidden.append({"claim": "say_no_new_obligation_when_added", "obligations": delta["obligations_added"]})
    if delta["result_type"] == "unaffected": forbidden.append({"claim": "claim_update_changed_this_case"})
    basis = sorted(set(new["basis"]))
    return {"case_family_id": family_id, "event_id": event["event_id"], "domain_id": dom["domain_id"],
            "result_type": delta["result_type"], "must_satisfy": required,
            "acceptable_expression": {"answer_format": "free text", "semantic_equivalence": "any wording preserving must_satisfy and not asserting forbidden claims",
                                     "decision_label_set": [new["final_decision"]], "handling_label_set": [new["current_handling"]]},
            "field_definitions": {"human_review": HUMAN_REVIEW_SCOPE},
            "forbidden_claims": forbidden, "reasoning_basis": basis,
            "reasoning_requirements": {"must_explain": basis, "must_link_rule_to_case": True,
                                       "unsupported_inference_is_disallowed": True,
                                       "do_not_claim_update_effect_if_result_type_is_unaffected": delta["result_type"] == "unaffected"},
            "evidence_available": [{"evidence_id": x["evidence_id"], "supports_fact_keys": x["fact_keys"]} for x in evidence],
            "old_state": old, "new_state": new, "delta": delta,
            "no_unique_reference_essay": True}


def render_constraint_answer(state_now: dict[str, Any], delta: dict[str, Any]) -> str:
    def items(values: list[str]) -> str: return "、".join(values) if values else "无"
    changed = delta["obligation_status_changed"]
    status_text = "；".join(f"{k}：{v['old']}→{v['new']}" for k, v in changed.items()) or "无既有义务状态变化"
    review = state_now["human_review"]
    return (f"最终决策状态：{state_now['final_decision']}。当前处置：{state_now['current_handling']}。"
            f"新增义务：{items(delta['obligations_added'])}；取消义务：{items(delta['obligations_removed'])}；"
            f"保持义务：{items(delta['obligations_unchanged'])}。义务状态变化：{status_text}。"
            f"新增程序：{items(delta['procedures_added'])}；取消程序：{items(delta['procedures_removed'])}。"
            f"允许事项：{items(state_now['permissions'])}；禁止事项：{items(state_now['prohibitions'])}。"
            f"人工复核：{'需要' if review['required'] else '不需要'}，状态为{review['status']}。"
            f"理由依据：{items(state_now['basis'])}。")


def calibration_facts(event: dict[str, Any], dom: dict[str, Any]) -> dict[str, Any]:
    effective = date.fromisoformat(event["effective_date"])
    action = effective + timedelta(days=180)
    f: dict[str, Any] = {"action_date": action.isoformat()}
    kind = event["kind"]
    if kind == "age_threshold":
        f.update({"birth_date": birth_for_age(19, action.isoformat(), True), "secondary_gate_met": True})
    elif kind == "scope_expansion":
        f.update({"feature_type": "voiceprint", "identity_inference": True, "identity_feature_generated": True,
                  "separate_consent": False, "encrypted": True, "purpose_matches_domain": True, "retention_days": 14})
    elif kind == "review_strengthening":
        impact_key = dom["impact_fact"] or "high_impact"
        f.update({impact_key: True, "adverse_proposed": True, "formal_confirmation": True,
                  "materials_checked": True, "system_reason_checked": True, "reviewer_can_override": False,
                  "review_reason_recorded": True, "domain_file_access": True})
    elif kind == "validity_window_shortening": f.update({"days_since_last_check": 120, "secondary_gate_met": True})
    elif kind == "evidence_threshold_increase":
        f.update({"independent_source_count": 1, "core_claim_true": True, "evidence_sources": ["校准独立来源A"]})
        f[dom["secondary_gate"]] = True
    elif kind == "permission_default_reversal": f.update({"affirmative_authorization": False, "explicit_objection": False, "authorization_scope_matches": True})
    elif kind == "exception_narrowing": f.update({"restricted_condition": True, "urgent": True, "immediate_safety_risk": False,
                                                    "safe_alternative_available": False, "qualified_responder_available": False})
    elif kind == "conflict_priority_change": f.update({"upper_safety_restriction_applies": True, "local_exception_allows_action": True,
                                                        "direct_conflict": True, "independent_safeguard_complete": False})
    elif kind == "transition_scope_expansion":
        f.update({"filed_date": (effective - timedelta(days=1)).isoformat(), "request_status": "pending",
                  "legacy_requirement_met": True})
        for k in dom["secondary_gate"].split("+"):
            f[k] = k != "return_contact_plan_present"
    else: raise ValueError(kind)
    return f


def build_scorer_calibration() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    challenges, hidden_keys = [], []
    for event in EVENTS:
        dom = next(d for d in DOMAINS[event["event_id"]] if d["split"] == "train")
        facts = calibration_facts(event, dom)
        old, new = evaluate(event, dom, facts, "old"), evaluate(event, dom, facts, "new")
        assert_state_matches_oracle(event, dom, facts, "old")
        assert_state_matches_oracle(event, dom, facts, "new")
        delta = make_delta(old, new, event, dom, facts)
        family = f"CAL-{event['event_id']}"
        text, evidence = render_case(event, dom, facts, "hard")
        prompt = prompt_text(event, dom, text, "ice")
        correct = render_constraint_answer(new, delta)
        forbidden_name = "、".join(new["prohibitions"]) or "无"
        permitted_name = "、".join(new["permissions"]) or "无"
        if new["prohibitions"]:
            wrong_action = correct + f" 另外，允许事项：{forbidden_name}。"
        else:
            wrong_action = correct + f" 另外，禁止事项：{permitted_name}。"
        action_name = dom["action"]
        contradiction = correct + f" 同一行为既允许又禁止：{action_name}；既应执行又不得执行。"
        wrong_status = correct + " 但我另行断言：上述新增义务和保持义务均无需履行，当前所有义务也都已完成。"
        challenge_id = "SCORER-" + sha256_bytes(f"{SEED}:{family}".encode())[:16]
        challenges.append({"challenge_id": challenge_id, "split": "scorer_calibration_only", "prompt": prompt,
                           "candidate_responses": [
                               {"response_id": "positive", "text": correct},
                               {"response_id": "wrong_permission", "text": wrong_action},
                               {"response_id": "contradiction", "text": contradiction},
                               {"response_id": "wrong_obligation_status", "text": wrong_status},
                           ]})
        hidden_keys.append({"challenge_id": challenge_id, "event_id": event["event_id"], "domain_id": dom["domain_id"],
                            "facts": facts, "old_state": old, "new_state": new, "delta": delta,
                            "expected_candidate_verdicts": {"positive": {"pass": True, "fail_fields": []},
                                                             "wrong_permission": {"pass": False, "fail_fields": ["permissions", "prohibitions", "internal_consistency"]},
                                                             "contradiction": {"pass": False, "fail_fields": ["permissions", "prohibitions", "internal_consistency"]},
                                                             "wrong_obligation_status": {"pass": False, "fail_fields": ["obligations_added", "obligations_unchanged", "obligation_status"]}},
                            "evidence": evidence, "not_for_model_training_or_main_evaluation": True})
    return challenges, hidden_keys


def prompt_text(event: dict[str, Any], dom: dict[str, Any], case_text: str, condition: str) -> str:
    header = "以下均为虚构受控研究环境规则，不代表现实法律、政策或专业意见。\n"
    historic = f"历史上层规则（仅用于比较，不代表本案现行版本）：{event['old_rule']}\n"
    domain = f"本领域规则：{domain_rule_text(event, dom)}\n"
    update = f"当前更新后的上层规则：{event['new_rule']}\n" if condition == "ice" else ""
    query = ("案例材料：\n" + case_text + "\n"
             + HUMAN_REVIEW_SCOPE + "\n"
             "请依据本案应适用的规则，直接说明：最终决定、当前处置、应新增/取消/保持的义务、程序要求，以及按上述口径是否需要人工复核；解释关键事实和规则如何共同支持结论。若材料不足以推出某项结论，请明确指出缺失条件。")
    return header + historic + domain + update + query


def assert_state_matches_oracle(event: dict[str, Any], dom: dict[str, Any], facts: dict[str, Any], version: str,
                                exception_override: bool | None = None) -> None:
    actual = evaluate(event, dom, facts, version, exception_override)
    expected = oracle_full_state(event, dom, facts, version, exception_override)
    if actual != expected:
        differences = {key: {"engine": actual.get(key), "oracle": expected.get(key)}
                       for key in set(actual) | set(expected) if actual.get(key) != expected.get(key)}
        raise AssertionError({"event": event["event_id"], "domain": dom["domain_id"], "facts": facts,
                              "version": version, "differences": differences})


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows), encoding="utf-8")


def validate_dataset(root: Path, payload: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    cases = payload["cases"]
    gold = payload["gold"]
    facts = payload["facts"]
    prompts_cb = payload["prompts_closed_book"]
    prompts_ice = payload["prompts_ice"]
    domains = payload["domains"]
    calibration = payload["calibration_challenges"]
    calibration_gold = payload["calibration_gold"]
    train_inputs = payload["train_inputs"]
    train_targets = payload["train_targets"]
    events = {e["event_id"]: e for e in EVENTS}
    errors: list[str] = []
    supported_validity_modes = {"validity_hold", "validity_maintenance", "validity_reconsent",
                                "validity_quarantine", "validity_downgrade", "validity_route"}
    validity_domains = [d for d in domains if d["event_id"] == "RTE_VALIDITY_180_TO_90"]
    if any(d["mode"] not in supported_validity_modes for d in validity_domains):
        errors.append("unsupported_validity_domain_mode")
    ids = [x["request_id"] for x in prompts_cb]
    errors += ["duplicate_request_id"] if len(ids) != len(set(ids)) else []
    if any(HUMAN_REVIEW_SCOPE not in row["text"] for row in prompts_cb + prompts_ice):
        errors.append("human_review_scope_missing_from_prompt")
    if any(row.get("field_definitions", {}).get("human_review") != HUMAN_REVIEW_SCOPE for row in gold):
        errors.append("human_review_scope_missing_from_gold")
    expected_domains = len(EVENTS) * sum(BASE_SPLITS.values()) + sum(TEST_DOMAINS_PER_EVENT.values())
    expected_families = expected_domains * N_FAMILIES_PER_DOMAIN
    expected_cases = expected_families * len(VARIANTS)
    expected_train_families = len(EVENTS) * BASE_SPLITS["train"] * N_FAMILIES_PER_DOMAIN
    if len(cases) != expected_cases: errors.append(f"case_rows={len(cases)}")
    if len(facts) != expected_families: errors.append(f"families={len(facts)}")
    if len(gold) != expected_families: errors.append(f"gold={len(gold)}")
    if len(calibration) != len(EVENTS) or len(calibration_gold) != len(EVENTS): errors.append("scorer_calibration_coverage")
    if any(len(x["candidate_responses"]) != 4 for x in calibration): errors.append("scorer_calibration_candidate_count")
    if {x["challenge_id"] for x in calibration} != {x["challenge_id"] for x in calibration_gold}: errors.append("scorer_calibration_gold_join")
    if len(prompts_cb) != len(prompts_ice) or len(prompts_cb) != len(cases): errors.append("prompt_pair_coverage")
    if len(train_inputs) != expected_train_families * len(VARIANTS) or len(train_targets) != expected_train_families * len(VARIANTS): errors.append("train_input_target_count")
    if {x["request_id"] for x in train_inputs} != {x["request_id"] for x in train_targets}: errors.append("train_input_target_join")
    if any(x["split"] != "train" for x in train_inputs + train_targets): errors.append("non_train_row_in_training_files")
    new_rule_by_event = {e["event_id"]: e["new_rule"] for e in EVENTS}
    if any(new_rule_by_event[x["event_id"]] not in x["text"] for x in train_inputs): errors.append("missing_updated_rule_in_train_input")
    case_event_by_request = {x["request_id"]: x["event_id"] for x in cases}
    expected_train_ids = {x["request_id"] for x in cases if x["split"] == "train"}
    if {x["request_id"] for x in train_inputs} != expected_train_ids: errors.append("train_contains_missing_or_nontrain_cases")
    split_public = payload["split_prompts"]
    case_split_by_request = {x["request_id"]: x["split"] for x in cases}
    for split, family_count in (("train", expected_train_families),
                                ("development", len(EVENTS) * BASE_SPLITS["development"] * N_FAMILIES_PER_DOMAIN),
                                ("test", EXPECTED_TEST_FAMILIES)):
        expected_ids = {request_id for request_id, assigned_split in case_split_by_request.items() if assigned_split == split}
        for condition in ("closed_book", "ice"):
            split_rows = split_public[f"requests_{split}_{condition}.jsonl"]
            if len(split_rows) != family_count * len(VARIANTS): errors.append(f"split_request_count:{split}:{condition}")
            if {x["request_id"] for x in split_rows} != expected_ids: errors.append(f"split_request_join:{split}:{condition}")
            if any(set(x) != {"request_id", "text"} for x in split_rows): errors.append(f"split_request_metadata_leak:{split}:{condition}")
    train_family_by_request = {x["request_id"]: x["case_family_id"] for x in train_inputs}
    if any(train_family_by_request.get(x["request_id"]) != x["case_family_id"] for x in train_targets): errors.append("train_target_family_mismatch")
    by_event_domain = Counter((x["event_id"], x["domain_id"], x["split"]) for x in facts)
    event_splits = defaultdict(Counter)
    event_domain_chains = defaultdict(set)
    event_fact_signatures: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in facts: event_fact_signatures[row["event_id"]].append(row)
    for event_id, rows in event_fact_signatures.items():
        signatures = [x["scenario_hash"] for x in rows]
        if len(set(signatures)) != len(signatures): errors.append(f"duplicate_fact_vector_across_event:{event_id}")
        split_signatures = {split: {x["scenario_hash"] for x in rows if x["split"] == split}
                            for split in ("train", "development", "test")}
        if split_signatures["train"] & split_signatures["development"]: errors.append(f"fact_vector_train_dev_overlap:{event_id}")
        if split_signatures["train"] & split_signatures["test"]: errors.append(f"fact_vector_train_test_overlap:{event_id}")
        if split_signatures["development"] & split_signatures["test"]: errors.append(f"fact_vector_dev_test_overlap:{event_id}")
    for d in domains:
        if d["domain_id"] in HISTORIC_DOMAINS: errors.append(f"historical_domain_in_new_data:{d['domain_id']}")
        event_splits[d["event_id"]][d["split"]] += 1
        event_domain_chains[d["event_id"]].add(d["decision_chain"])
    for e in EVENTS:
        expected_event_split = Counter({"train": BASE_SPLITS["train"], "development": BASE_SPLITS["development"],
                                        "test": TEST_DOMAINS_PER_EVENT[e["event_id"]]})
        if event_splits[e["event_id"]] != expected_event_split:
            errors.append(f"domain_split_quota:{e['event_id']}:{event_splits[e['event_id']]}")
        if len(event_domain_chains[e["event_id"]]) != sum(expected_event_split.values()):
            errors.append(f"domain_decision_chain_count:{e['event_id']}")
    family_variants = defaultdict(list)
    for c in cases: family_variants[c["case_family_id"]].append(c)
    if len(family_variants) != expected_families: errors.append("family_id_count")
    if any(Counter(x["difficulty"] for x in v) != Counter({"easy": 1, "medium": 1, "hard": 1}) for v in family_variants.values()):
        errors.append("variant_count_or_split")
    if any(len({x["split"] for x in v}) != 1 or len({x["domain_id"] for x in v}) != 1 or len({x["scenario_hash"] for x in v}) != 1 for v in family_variants.values()):
        errors.append("family_leakage_or_variant_fact_drift")
    if any(len({x["scenario_hash"] for x in facts if x["domain_id"] == d["domain_id"]}) != N_FAMILIES_PER_DOMAIN for d in domains):
        errors.append("duplicate_scenario_within_domain")
    cb = {x["request_id"]: x for x in prompts_cb}
    ice = {x["request_id"]: x for x in prompts_ice}
    if set(cb) != set(ice): errors.append("ice_closed_book_id_mismatch")
    if set(cb) != {x["request_id"] for x in cases}: errors.append("request_case_join_coverage")
    hidden_terms = ("decision_changed", "procedure_only", "exception_preserves", "unaffected", "easy", "medium", "hard", "case_family_id", "gold")
    for request_id, row in cb.items():
        text = row["text"]
        if any(t in text for t in hidden_terms): errors.append(f"hidden_label_in_prompt:{request_id}")
        if any(e["new_rule"] in text for e in EVENTS): errors.append(f"new_rule_leak_in_closed_book:{request_id}")
        if request_id not in ice or not any(e["new_rule"] in ice[request_id]["text"] for e in EVENTS): errors.append(f"missing_ice_rule:{request_id}")
        event_rule = new_rule_by_event[case_event_by_request[request_id]]
        if event_rule not in ice[request_id]["text"]: errors.append(f"wrong_ice_rule_for_event:{request_id}")
        if any(e["new_rule"] in ice[request_id]["text"] for e in EVENTS if e["new_rule"] != event_rule):
            errors.append(f"cross_event_rule_leak_in_ice:{request_id}")
    if len({d["domain_id"] for d in domains}) != expected_domains: errors.append("domain_id_uniqueness")
    chain_signatures = defaultdict(set)
    for d in domains:
        signature = json.dumps({k:d[k] for k in ("mode", "decision_chain", "secondary_gate", "fallback_action", "invariant_obligation", "local_rule")}, ensure_ascii=False, sort_keys=True)
        if signature in chain_signatures[d["event_id"]]: errors.append(f"duplicate_domain_chain:{d['event_id']}:{d['domain_id']}")
        chain_signatures[d["event_id"]].add(signature)
    for g in gold:
        event = events[g["event_id"]]
        dom = next(d for d in domains if d["domain_id"] == g["domain_id"])
        f = next(x["facts"] for x in facts if x["case_family_id"] == g["case_family_id"])
        if event["kind"] == "scope_expansion" and f["identity_feature_generated"] and not f["identity_inference"]:
            errors.append(f"biometric_fact_contradiction:{g['case_family_id']}")
        if event["kind"] == "permission_default_reversal":
            if f["affirmative_authorization"] and (not f["authorization_record_exists"] or not f["authorization_scope_matches"] or f["explicit_objection"]):
                errors.append(f"permission_fact_contradiction:{g['case_family_id']}")
            if (f["affirmative_authorization"] or f["explicit_objection"]) and not f["response_received"]:
                errors.append(f"permission_response_missing:{g['case_family_id']}")
            if not f["authorization_record_exists"] and not f["authorization_scope_matches"]:
                errors.append(f"authorization_scope_without_record:{g['case_family_id']}")
        if event["kind"] == "exception_narrowing" and f["immediate_safety_risk"] and not f["urgent"]:
            errors.append(f"immediate_risk_without_urgency:{g['case_family_id']}")
        if event["kind"] == "conflict_priority_change" and f["direct_conflict"] and not (f["upper_safety_restriction_applies"] and f["local_exception_allows_action"]):
            errors.append(f"direct_conflict_fact_inconsistent:{g['case_family_id']}")
        if event["kind"] == "transition_scope_expansion" and date.fromisoformat(f["filed_date"]) > date.fromisoformat(f["action_date"]):
            errors.append(f"transition_filed_after_action:{g['case_family_id']}")
        assert_state_matches_oracle(event, dom, f, "old")
        assert_state_matches_oracle(event, dom, f, "new")
        expected_old = oracle_full_state(event, dom, f, "old")
        expected_new = oracle_full_state(event, dom, f, "new")
        if g["old_state"] != expected_old: errors.append(f"old_state_gold_mismatch:{g['case_family_id']}")
        if g["new_state"] != expected_new: errors.append(f"new_state_gold_mismatch:{g['case_family_id']}")
        recomputed = oracle_delta(expected_old, expected_new, event, dom, f)
        if recomputed != g["delta"]: errors.append(f"full_delta_mismatch:{g['case_family_id']}")
        if event["kind"] == "evidence_threshold_increase" and len(f["evidence_sources"]) != f["independent_source_count"]:
            errors.append(f"evidence_count_mismatch:{g['case_family_id']}")
        if g["delta"]["result_type"] == "exception_preserves":
            counterfactual_old = oracle_full_state(event, dom, f, "old", exception_override=False)
            counterfactual_new = oracle_full_state(event, dom, f, "new", exception_override=False)
            if not active_new(event, f, "new") or g["old_state"] != g["new_state"] or counterfactual_old == counterfactual_new:
                errors.append(f"exception_counterfactual_failed:{g['case_family_id']}")
        if event["kind"] == "transition_scope_expansion":
            required_transition_facts = set(dom["secondary_gate"].split("+"))
            if "new_requirement_met" in f: errors.append(f"transition_aggregate_leak:{g['case_family_id']}")
            if f["request_status"] == "pending" and not required_transition_facts.issubset(f):
                errors.append(f"transition_checklist_missing:{g['case_family_id']}")
            if f["request_status"] == "pending" and date.fromisoformat(f["filed_date"]) < date.fromisoformat(event["effective_date"]) and "legacy_requirement_met" not in f:
                errors.append(f"transition_legacy_check_missing:{g['case_family_id']}")
            if f["request_status"] == "executed" and (required_transition_facts & set(f)):
                errors.append(f"transition_irrelevant_checklist:{g['case_family_id']}")
        if event["kind"] == "transition_scope_expansion" and f["request_status"] == "executed" and g["new_state"]["current_handling"] != "preserve_completed_action":
            errors.append(f"retroactive_transition_error:{g['case_family_id']}")
    evidence_by_request = {row["request_id"]: row for row in payload["evidence"]}
    facts_by_family = {row["case_family_id"]: row["facts"] for row in facts}
    cases_by_request = {row["request_id"]: row for row in cases}
    for family, variants in family_variants.items():
        expected_keys = set(facts_by_family[family])
        for variant in variants:
            evidence = evidence_by_request[variant["request_id"]]["case_text_evidence"]
            exposed_keys = {key for segment in evidence for key in segment["fact_keys"]}
            if exposed_keys != expected_keys: errors.append(f"variant_fact_coverage:{variant['request_id']}")
            case_text = cases_by_request[variant["request_id"]]["case_text"]
            for segment in evidence:
                start, end = segment["char_start"], segment["char_end"]
                if start < 0 or end < start or case_text[start:end] != segment["text"]:
                    errors.append(f"evidence_span_mismatch:{variant['request_id']}:{segment['evidence_id']}")
    for event in EVENTS:
        event_facts = [row["facts"] for row in facts if row["event_id"] == event["event_id"]]
        event_date = date.fromisoformat(event["effective_date"])
        if not any(date.fromisoformat(row["action_date"]) < event_date for row in event_facts): errors.append(f"no_pre_effective_cases:{event['event_id']}")
        if not any(date.fromisoformat(row["action_date"]) >= event_date for row in event_facts): errors.append(f"no_post_effective_cases:{event['event_id']}")
    if errors: raise AssertionError("Validation failed: " + "; ".join(errors[:40]))
    result_types = Counter(g["delta"]["result_type"] for g in gold)
    by_event_types = {e["event_id"]: dict(Counter(g["delta"]["result_type"] for g in gold if g["event_id"] == e["event_id"])) for e in EVENTS}
    split_by_family = {x["case_family_id"]: x["split"] for x in facts}
    by_split_types = {split: dict(Counter(g["delta"]["result_type"] for g in gold if split_by_family[g["case_family_id"]] == split))
                      for split in ("train", "development", "test")}
    if by_split_types["test"].get("exception_preserves", 0) < 6:
        raise AssertionError("test_exception_preserves_coverage_below_minimum")
    return {"status": "passed", "checks": {"unique_request_ids": len(ids), "case_rows": len(cases), "semantic_families": len(family_variants), "variants_per_family": 3,
                 "domains": len(domains), "events": len(EVENTS), "train_families": sum(x["split"]=="train" for x in facts),
                 "development_families": sum(x["split"]=="development" for x in facts), "test_families": sum(x["split"]=="test" for x in facts),
                 "new_rule_absent_from_closed_book": True, "full_state_forward_oracle_agreement": 2 * len(facts),
                 "full_delta_independent_recompute": len(gold), "evidence_spans_exact": len(evidence_by_request),
                 "train_input_target_pairs": len(train_inputs), "training_contains_new_rule": True,
                 "historical_domains_excluded": True, "unique_chain_profiles_per_event": True,
                 "test_domains": sum(TEST_DOMAINS_PER_EVENT.values()), "test_families_target": EXPECTED_TEST_FAMILIES,
                 "split_specific_prompt_files": {split: {condition: len(payload["split_prompts"][f"requests_{split}_{condition}.jsonl"])
                                                          for condition in ("closed_book", "ice")}
                                                   for split in ("train", "development", "test")}},
                 "result_type_counts": dict(result_types), "result_type_counts_by_event": by_event_types,
            "result_type_counts_by_split": by_split_types,
            "validity_modes_supported": True,
            "limitations": ["Mechanical agreement verifies implementation consistency, not the real-world validity of the invented rules.",
                            "The 1,700 test-family target is a planning estimate for a 5pp MDE under assumed domain ICC=.05; it is not an empirical power guarantee or frozen sample size.",
                            "Surface forms are controlled templates; automatic fact equivalence does not prove arbitrary paraphrase equivalence."]}


def generate(output: Path, force: bool = False) -> dict[str, Any]:
    safe_rebuild = False
    if output.exists() and any(output.iterdir()):
        marker = output / "manifest.json"
        if not force:
            raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
        if marker.exists():
            prior = json.loads(marker.read_text(encoding="utf-8"))
            if prior.get("version") != VERSION or prior.get("status") not in {
                "draft_not_frozen", "candidate_pending_free_prose_scorer_validation",
                "generated_pending_finalization",
            }:
                raise FileExistsError(f"Refusing to replace output not owned by this draft generator: {output}")
            safe_rebuild = True
        elif {p.name for p in output.iterdir()} <= {"README.md", "POWER_ESTIMATE.md", "SCORING_PROTOCOL.md", "DATASET_AUDIT.md", "sample_size_sensitivity.ipynb"}:
            safe_rebuild = True
        else:
            raise FileExistsError(f"Refusing to overwrite non-empty output without this generator's manifest: {output}")
    output.mkdir(parents=True, exist_ok=True)
    all_domains = [d for e in EVENTS for d in DOMAINS[e["event_id"]]]
    expected_domains = len(EVENTS) * sum(BASE_SPLITS.values()) + sum(TEST_DOMAINS_PER_EVENT.values())
    if len(all_domains) != expected_domains: raise AssertionError((len(all_domains), expected_domains))
    for e in EVENTS:
        event_expected = {**BASE_SPLITS, "test": TEST_DOMAINS_PER_EVENT[e["event_id"]]}
        if len(DOMAINS[e["event_id"]]) != sum(event_expected.values()): raise AssertionError(e["event_id"])
        for split, expected in event_expected.items():
            actual = sum(d["split"] == split for d in DOMAINS[e["event_id"]])
            if actual != expected: raise AssertionError((e["event_id"], split, actual))
    rules = EVENTS
    domain_rows = [{**d, "event_id": e["event_id"], "domain_rule": domain_rule_text(e, d),
                    "chain_signature": {k:d[k] for k in ("mode", "decision_chain", "secondary_gate", "fallback_action", "invariant_obligation", "local_rule")}}
                   for e in EVENTS for d in DOMAINS[e["event_id"]]]
    facts_rows, case_rows, gold_rows, cb_rows, ice_rows, evidence_rows = [], [], [], [], [], []
    event_by_id = {e["event_id"]: e for e in EVENTS}
    for event in EVENTS:
        for dom in DOMAINS[event["event_id"]]:
            rng = random.Random(stable_seed(dom["domain_id"]))
            for i in range(N_FAMILIES_PER_DOMAIN):
                family = f"FAM-{event['event_id']}-{dom['domain_id']}-{i+1:02d}"
                f = generate_facts(event, dom, i, rng)
                old = evaluate(event, dom, f, "old")
                new = evaluate(event, dom, f, "new")
                assert_state_matches_oracle(event, dom, f, "old")
                assert_state_matches_oracle(event, dom, f, "new")
                delta = make_delta(old, new, event, dom, f)
                family_payload = {"case_family_id": family, "event_id": event["event_id"], "domain_id": dom["domain_id"],
                                  "domain_name": dom["name"], "split": dom["split"], "facts": f,
                                  "old_state": old, "new_state": new, "delta": delta,
                                  "scenario_hash": sha256_bytes(json.dumps(f, sort_keys=True, ensure_ascii=False).encode())}
                facts_rows.append(family_payload)
                for difficulty in VARIANTS:
                    case_text, evidence = render_case(event, dom, f, difficulty)
                    # Request IDs are opaque and do not reveal split, event, domain, or difficulty.
                    request_id = "REQ-" + sha256_bytes(f"{SEED}:{family}:{difficulty}".encode())[:20]
                    cb = prompt_text(event, dom, case_text, "closed_book")
                    ice = prompt_text(event, dom, case_text, "ice")
                    case_rows.append({"request_id": request_id, "case_family_id": family, "difficulty": difficulty,
                                      "event_id": event["event_id"], "domain_id": dom["domain_id"], "split": dom["split"],
                                      "scenario_hash": family_payload["scenario_hash"], "case_text": case_text})
                    cb_rows.append({"request_id": request_id, "text": cb})
                    ice_rows.append({"request_id": request_id, "text": ice})
                    evidence_rows.append({"request_id": request_id, "case_family_id": family,
                                          "difficulty": difficulty, "case_text_evidence": evidence,
                                          "facts_hash": family_payload["scenario_hash"]})
                evidence = render_case(event, dom, f, "easy")[1]
                gold_rows.append(case_gold(family, event, dom, f, old, new, delta, evidence))
    calibration_challenges, calibration_gold = build_scorer_calibration()
    gold_by_family = {row["case_family_id"]: row for row in gold_rows}
    ice_by_request = {row["request_id"]: row for row in ice_rows}
    evidence_by_request = {row["request_id"]: row for row in evidence_rows}
    split_by_family = {row["case_family_id"]: row["split"] for row in facts_rows}
    train_inputs, train_targets = [], []
    for case in case_rows:
        if split_by_family[case["case_family_id"]] != "train":
            continue
        request_id = case["request_id"]
        family_gold = gold_by_family[case["case_family_id"]]
        train_inputs.append({"request_id": request_id, "case_family_id": case["case_family_id"],
                             "event_id": case["event_id"], "domain_id": case["domain_id"],
                             "split": "train", "text": ice_by_request[request_id]["text"]})
        train_targets.append({"request_id": request_id, "case_family_id": case["case_family_id"],
                              "event_id": case["event_id"], "domain_id": case["domain_id"], "split": "train",
                              "target_constraints": family_gold["must_satisfy"],
                              "forbidden_claims": family_gold["forbidden_claims"],
                              "reasoning_requirements": family_gold["reasoning_requirements"],
                              "old_state": family_gold["old_state"], "new_state": family_gold["new_state"],
                              "delta": family_gold["delta"],
                              "evidence_available": evidence_by_request[request_id]["case_text_evidence"]})
    case_split_by_request = {row["request_id"]: row["split"] for row in case_rows}
    split_prompts = {}
    for split in ("train", "development", "test"):
        split_prompts[f"requests_{split}_closed_book.jsonl"] = [row for row in cb_rows if case_split_by_request[row["request_id"]] == split]
        split_prompts[f"requests_{split}_ice.jsonl"] = [row for row in ice_rows if case_split_by_request[row["request_id"]] == split]
    payload = {"domains": domain_rows, "facts": facts_rows, "cases": case_rows, "gold": gold_rows,
               "prompts_closed_book": cb_rows, "prompts_ice": ice_rows, "evidence": evidence_rows,
               "train_inputs": train_inputs, "train_targets": train_targets,
               "calibration_challenges": calibration_challenges, "calibration_gold": calibration_gold,
               "split_prompts": split_prompts}
    validation = validate_dataset(output, payload)
    if safe_rebuild:
        generated_names = {
            "rules.jsonl", "domains.jsonl", "case_facts_gold.jsonl", "case_families.jsonl",
            "gold_constraints.jsonl", "cases.jsonl", "requests_closed_book.jsonl", "requests_ice.jsonl",
            "requests_train_closed_book.jsonl", "requests_train_ice.jsonl",
            "requests_development_closed_book.jsonl", "requests_development_ice.jsonl",
            "requests_test_closed_book.jsonl", "requests_test_ice.jsonl",
            "fact_evidence.jsonl", "scorer_calibration_challenges.jsonl", "scorer_calibration_key.jsonl",
            "requests_train_with_rule.jsonl", "targets_train_constraints.jsonl",
            "split_manifest.json", "validation_report.json", "manifest.json", "SHA256.json",
        }
        for name in generated_names:
            path = output / name
            if path.exists(): path.unlink()
    files = {
        "rules.jsonl": rules, "domains.jsonl": domain_rows, "case_facts_gold.jsonl": facts_rows,
        "case_families.jsonl": [{k:v for k,v in x.items() if k not in {"facts", "old_state", "new_state", "delta"}} for x in facts_rows],
        "gold_constraints.jsonl": gold_rows, "cases.jsonl": case_rows,
        "requests_closed_book.jsonl": cb_rows, "requests_ice.jsonl": ice_rows,
        **split_prompts,
        "fact_evidence.jsonl": evidence_rows,
        "scorer_calibration_challenges.jsonl": calibration_challenges,
        "scorer_calibration_key.jsonl": calibration_gold,
        "requests_train_with_rule.jsonl": train_inputs,
        "targets_train_constraints.jsonl": train_targets,
    }
    for name, rows in files.items(): write_jsonl(output / name, rows)
    split_manifest = {
        "policy": "domain-held-out by event; no case family or difficulty variant crosses splits",
        "events": {
            event["event_id"]: {
                split: [{"domain_id": d["domain_id"], "domain_name": d["name"], "case_families": N_FAMILIES_PER_DOMAIN}
                        for d in domain_rows if d["event_id"] == event["event_id"] and d["split"] == split]
                for split in ("train", "development", "test")
            }
            for event in EVENTS
        },
    }
    (output / "split_manifest.json").write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "validation_report.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    split_counts = Counter(x["split"] for x in facts_rows)
    manifest = {
        "version": VERSION, "status": "generated_pending_finalization", "generation_seed": SEED,
        "generation": {"generator": "deterministic Python controlled templates", "model_generated_gold": False,
                       "human_case_labeling": False, "gold_source": "event/domain executable state transitions plus separate full-state direct oracle and set-difference recomputation",
                       "surface_generation": "three controlled evidence presentations from identical typed facts", "rule_engine_version": "rule-transition-main-v1",
                       "scenario_randomization": "seeded domain-specific stratified permutations; event-wide unique action dates, approximately 45% pre-effective and 55% effective-or-later; exact fact vectors are unique within each event and split-disjoint",
                       "independent_oracle_scope": "old/new decisions, handling, obligations, procedures, permissions, prohibitions, human review and basis; deltas independently recomputed",
                       "generator_sha256": sha256_bytes(Path(__file__).read_bytes()), "python_version": platform.python_version()},
        "counts": {"events": len(EVENTS), "domains": len(domain_rows), "semantic_case_families": len(facts_rows),
                   "language_variants": len(case_rows), "closed_book_prompts": len(cb_rows), "ice_prompts": len(ice_rows),
                   "total_conditioned_inputs": len(cb_rows) + len(ice_rows), "scorer_calibration_cases": len(calibration_challenges),
                   "scorer_calibration_candidate_responses": sum(len(x["candidate_responses"]) for x in calibration_challenges),
                   "train_inputs_with_updated_rule": len(train_inputs), "train_structured_targets": len(train_targets),
                   "train_families": split_counts["train"], "development_families": split_counts["development"], "test_families": split_counts["test"],
                   "train_prompts": split_counts["train"]*3, "development_prompts": split_counts["development"]*3, "test_prompts": split_counts["test"]*3,
                   "test_closed_book_prompts": len(split_prompts["requests_test_closed_book.jsonl"]),
                   "test_ice_prompts": len(split_prompts["requests_test_ice.jsonl"])},
        "split_policy": "domain-held-out within every event: 3 train, 1 development, and 9-10 unseen-test domains per event; all three language forms remain within family split",
        "historical_data_policy": {"legacy_case_or_gold_loaded": False, "historic_twelve_domains_used": "exclusion-list only", "historic_domains_in_new_benchmark": 0,
                                   "round34_pilot_used_for": "sample-size estimation only; no pilot cases, prompts, facts, or labels copied"},
        "outcome_counts": validation["result_type_counts"], "outcome_counts_by_event": validation["result_type_counts_by_event"],
        "sample_size_status": "planning_target_only_not_power_frozen",
        "sample_size_note": "5pp MDE planning target: 1,700 test case families across 85 held-out domains at assumed ICC=.05; estimate uses pilot paired variance that was method-selection biased and is not a formal power guarantee.",
        "human_review_scope": HUMAN_REVIEW_SCOPE,
        "scorer_status": "v0.9 passed a small mostly structured calibration but does not establish natural free-text reliability; separate unlabeled natural-prose calibration is pending and main test answers must not be used to tune the scorer.",
        "limitations": validation["limitations"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    hashes = {p.name: sha256_bytes(p.read_bytes()) for p in sorted(output.iterdir()) if p.is_file()}
    (output / "SHA256.json").write_text(json.dumps(hashes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"manifest": manifest, "validation": validation, "hashes": hashes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("新课题/data/main_experiment_v2_draft_20261012"))
    parser.add_argument("--force", action="store_true", help="replace only an existing output marked by this draft generator")
    args = parser.parse_args()
    result = generate(args.output, force=args.force)
    print(json.dumps({"output": str(args.output), "counts": result["manifest"]["counts"],
                      "outcomes": result["validation"]["result_type_counts"], "status": result["manifest"]["status"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
