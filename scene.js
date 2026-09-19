import * as THREE from './three.module.min.js';

// A real geometric model: no external asset service, image, or API dependency.
// Kept separate so a WebGL failure cannot affect navigation or applications.
document.querySelectorAll('[data-scene]').forEach(host => {
  const mount=host.querySelector('.scene-canvas'), fallback=host.querySelector('.scene-fallback');
  const reduced=matchMedia('(prefers-reduced-motion: reduce)');
  let renderer;
  try {
    renderer=new THREE.WebGLRenderer({alpha:true,antialias:true,powerPreference:'low-power'});
  } catch {
    host.querySelector('.scene-instruction').textContent='The equity house · architectural illustration';
    host.querySelector('.scene-controls').hidden=true; mount.hidden=true; return;
  }
  renderer.setPixelRatio(Math.min(devicePixelRatio,1.6));
  renderer.shadowMap.enabled=true; renderer.shadowMap.type=THREE.PCFSoftShadowMap;
  renderer.outputColorSpace=THREE.SRGBColorSpace;
  renderer.toneMapping=THREE.ACESFilmicToneMapping; renderer.toneMappingExposure=1.45;
  mount.append(renderer.domElement); fallback.hidden=true;
  const scene=new THREE.Scene();
  const camera=new THREE.OrthographicCamera(-3,3,2.3,-2.3,.1,50);
  camera.position.set(6,4.8,7);camera.lookAt(0,.87,0);
  scene.add(new THREE.HemisphereLight(0xfff8e7,0x6d7881,2.8));
  const sun=new THREE.DirectionalLight(0xfff5d7,4.2);sun.position.set(-3,7,5);sun.castShadow=true;
  sun.shadow.mapSize.set(1024,1024);sun.shadow.camera.left=-5;sun.shadow.camera.right=5;sun.shadow.camera.top=5;sun.shadow.camera.bottom=-5;
  sun.shadow.normalBias=.035;sun.shadow.bias=-.0001;sun.shadow.radius=5;scene.add(sun);
  const fill=new THREE.DirectionalLight(0xffffff,1.7);fill.position.set(5,3,-4);scene.add(fill);
  const model=new THREE.Group();scene.add(model);
  const material=(color,roughness=.7,metalness=0)=>new THREE.MeshStandardMaterial({color,roughness,metalness});
  const ivory=material(0xe9e6d7),ivoryLight=material(0xf8f5e8),green=material(0x21304c,.48,.12),roofRib=material(0x36445c,.5,.15);
  const bronze=material(0xb4995c,.32,.65),glass=material(0x969b79,.22,.4),frame=material(0x27354b,.65),stone=material(0xbbbea7);
  const shrub=material(0x536677),darkShrub=material(0x3a495b);
  function box(w,h,d,x,y,z,mat,parent=model){
    const mesh=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),mat);mesh.position.set(x,y,z);mesh.castShadow=true;mesh.receiveShadow=true;parent.add(mesh);return mesh;
  }
  // Three offset architectural plates suggest layers of equity, not a numerical chart.
  box(3.6,.14,2.85,0,-.38,0,bronze);
  box(3.83,.15,3.06,0,-.13,0,ivory);
  box(3.75,.12,3.02,0,.12,0,ivoryLight);
  box(3.05,.04,2.4,0,.2,0,stone);
  box(2.17,1.38,1.65,0,.91,0,ivoryLight);
  // Front and rear triangular gables.
  const triangle=new THREE.Shape();triangle.moveTo(-1.085,0);triangle.lineTo(1.085,0);triangle.lineTo(0,.83);triangle.closePath();
  const gableGeometry=new THREE.ExtrudeGeometry(triangle,{depth:1.65,bevelEnabled:false});
  const gable=new THREE.Mesh(gableGeometry,ivoryLight);gable.position.set(0,1.6,-.825);gable.castShadow=true;model.add(gable);
  const angle=Math.atan2(.83,1.085);
  const leftRoof=box(1.58,.075,1.98,-.6,1.984,0,green);leftRoof.rotation.z=angle;
  const rightRoof=box(1.58,.075,1.98,.6,1.984,0,green);rightRoof.rotation.z=-angle;
  for(let i=-4;i<=4;i++){
    const a=box(1.56,.017,.018,-.6,2.026,i*.22,roofRib);a.rotation.z=angle;
    const b=box(1.56,.017,.018,.6,2.026,i*.22,roofRib);b.rotation.z=-angle;
  }
  box(.075,.09,2.03,0,2.43,0,green);
  // Oak-toned window planes and deep green mullions.
  function windowFront(x,y,w,h){
    box(w+.065,h+.065,.05,x,y,.838,frame);box(w,h,.057,x,y,.846,glass);
    box(.027,h,.018,x,y,.888,ivory);box(w,.027,.018,x,y,.888,ivory);
    box(w+.1,.045,.11,x,y-h/2-.037,.87,ivory);
  }
  windowFront(-.46,1.00,.61,.63);
  windowFront(.44,1.95,.25,.27);
  box(.42,1.02,.05,.49,.73,.843,frame);
  box(.29,.41,.017,.49,.94,.878,glass);
  box(.045,.045,.035,.63,.67,.892,bronze);
  box(.62,.075,.4,.49,.237,1.00,ivoryLight);
  box(.78,.065,.27,.49,.182,1.3,ivory);
  for(const z of [-.42,.37]){
    box(.055,.69,.49,1.098,1.0,z,frame);box(.063,.6,.42,1.109,1.0,z,glass);
    box(.025,.61,.026,1.145,1.0,z,ivory);box(.025,.028,.42,1.145,1.0,z,ivory);
  }
  // Chimney, porch path, and modest planting give the model architectural character.
  box(.24,.67,.3,-.65,2.12,-.41,ivory);box(.3,.065,.35,-.65,2.46,-.41,bronze);
  for(let i=0;i<3;i++)box(.44,.025,.16,.49,.235,1.18+i*.17,ivoryLight);
  function plant(x,z,size,mat){
    box(size*1.28,.16,size*1.28,x,.31,z,ivory);
    const bush=new THREE.Mesh(new THREE.IcosahedronGeometry(size,1),mat);bush.scale.set(1,.85,1);bush.position.set(x,.45,z);bush.castShadow=true;model.add(bush);
  }
  plant(-1.4,.79,.26,darkShrub);plant(-1.38,-.02,.24,shrub);plant(1.39,-.93,.26,shrub);
  const treeTrunk=new THREE.Mesh(new THREE.CylinderGeometry(.035,.05,.55,8),bronze);treeTrunk.position.set(-1.3,.55,-.87);model.add(treeTrunk);
  const tree=new THREE.Mesh(new THREE.IcosahedronGeometry(.38,1),darkShrub);tree.scale.y=1.3;tree.position.set(-1.3,.99,-.87);tree.castShadow=true;model.add(tree);
  const ground=new THREE.Mesh(new THREE.PlaneGeometry(200,200),new THREE.ShadowMaterial({opacity:.12}));ground.rotation.x=-Math.PI/2;ground.position.y=-.51;ground.receiveShadow=true;scene.add(ground);
  let yaw=-.12,pitch=0,dragging=false,lastX=0,lastY=0,paused=reduced.matches,visible=true,alive=true,raf=null,start=performance.now();
  const pause=host.querySelector('[data-scene-pause]');
  function updatePause(){pause.setAttribute('aria-pressed',String(paused));pause.setAttribute('aria-label',paused?'Resume 3D motion':'Pause 3D motion');pause.title=paused?'Resume motion':'Pause motion';}
  updatePause();
  function render(time=performance.now()){
    model.rotation.y=yaw+(paused||dragging?0:Math.sin((time-start)*.00027)*.055);
    model.rotation.x=pitch;
    renderer.render(scene,camera);
  }
  function tick(t){raf=null;if(!alive||!visible||document.hidden)return;render(t);if(!paused)raf=requestAnimationFrame(tick);}
  function requestFrame(){if(!raf&&alive&&visible&&!document.hidden)raf=requestAnimationFrame(tick);}
  function resize(){
    const w=mount.clientWidth,h=mount.clientHeight;if(!w||!h)return;
    const aspect=w/h,frustum=host.classList.contains('large')?4.2:4.45;
    camera.left=-frustum*aspect/2;camera.right=frustum*aspect/2;camera.top=frustum/2;camera.bottom=-frustum/2;
    camera.updateProjectionMatrix();renderer.setSize(w,h,false);requestFrame();
  }
  const observer=new ResizeObserver(resize);observer.observe(mount);
  const visibility=new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;if(visible)requestFrame();else if(raf){cancelAnimationFrame(raf);raf=null;}},{threshold:0.01});visibility.observe(host);
  document.addEventListener('visibilitychange',()=>{if(document.hidden&&raf){cancelAnimationFrame(raf);raf=null;}else requestFrame();});
  mount.addEventListener('pointerdown',e=>{if(e.pointerType==='mouse'&&e.button!==0)return;dragging=true;lastX=e.clientX;lastY=e.clientY;mount.setPointerCapture(e.pointerId);mount.classList.add('dragging');});
  mount.addEventListener('pointermove',e=>{if(!dragging)return;yaw+=(e.clientX-lastX)*.008;pitch=THREE.MathUtils.clamp(pitch+(e.clientY-lastY)*.003,-.15,.22);lastX=e.clientX;lastY=e.clientY;requestFrame();});
  const stopDrag=()=>{dragging=false;mount.classList.remove('dragging');requestFrame();};
  mount.addEventListener('pointerup',stopDrag);mount.addEventListener('pointercancel',stopDrag);
  mount.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key))return;e.preventDefault();if(e.key==='ArrowLeft')yaw-=.14;if(e.key==='ArrowRight')yaw+=.14;if(e.key==='ArrowUp')pitch=Math.max(-.15,pitch-.06);if(e.key==='ArrowDown')pitch=Math.min(.22,pitch+.06);requestFrame();});
  host.querySelector('[data-scene-reset]').addEventListener('click',()=>{yaw=-.12;pitch=0;requestFrame();});
  pause.addEventListener('click',()=>{paused=!paused;updatePause();requestFrame();});
  reduced.addEventListener('change',e=>{paused=e.matches;updatePause();requestFrame();});
  renderer.domElement.addEventListener('webglcontextlost',e=>{e.preventDefault();alive=false;if(raf)cancelAnimationFrame(raf);fallback.hidden=false;mount.hidden=true;host.querySelector('.scene-controls').hidden=true;host.querySelector('.scene-instruction').textContent='The equity house · architectural illustration';});
  window.addEventListener('pagehide',()=>{alive=false;if(raf)cancelAnimationFrame(raf);observer.disconnect();visibility.disconnect();scene.traverse(obj=>{obj.geometry?.dispose();if(obj.material?.dispose)obj.material.dispose();});renderer.dispose();},{once:true});
  resize();requestFrame();
});
