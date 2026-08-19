const pages=[
['index.html','总入口'],['01_从零认识单细胞.html','01 单细胞基础'],['02_数据如何一步步变化.html','02 数据流'],
['03_数据集地图.html','03 数据集'],['04_SafeConf系统原理.html','04 系统原理'],['05_目录与代码结构.html','05 目录与代码'],
['06_实验怎么做.html','06 实验设计'],['07_结果与边界.html','07 结果边界'],['08_下一阶段任务.html','08 下一阶段'],['09_术语词典.html','09 术语词典'],
['10_代码设计与协议全解.html','10 协议与代码设计'],['11_实验结果完整谱系.html','11 实验证据谱系'],['12_真实文件逐项阅读.html','12 真实文件实习']
];
const nav=document.getElementById('chapterNav');
if(nav){const current=location.pathname.split('/').pop()||'index.html';nav.innerHTML='<div class="side-title">章节导航</div>'+pages.map(([u,t])=>`<a href="${u}" class="${u===current?'active':''}">${t}</a>`).join('')}
const input=document.getElementById('glossarySearch');
if(input){const terms=[...document.querySelectorAll('[data-term]')];input.addEventListener('input',()=>{const q=input.value.trim().toLowerCase();terms.forEach(x=>x.classList.toggle('hide',q&&!x.textContent.toLowerCase().includes(q)))})}
