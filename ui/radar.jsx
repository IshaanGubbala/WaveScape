// radar.jsx — Spatial threat radar (canvas + DOM hit overlay)

const { useRef, useEffect, useMemo, useState, useCallback } = React;

const DEFAULT_RADAR_RANGE_FT = 42;
const MIN_RADAR_RANGE_FT = 5;
const MAX_RADAR_RANGE_FT = 42;
const RADAR_RANGE_PAD = 1.35;
const M2FT = 3.28084;

function radarThreats(threats) {
  const valid = threats.filter(t => Number.isFinite(Number(t.distance)));
  if (valid.length <= 1) return valid;

  const strong = valid.filter(t => Number(t.confidence || 0) >= 20);
  const pool = strong.length ? strong : valid.slice().sort((a, b) => (b.confidence || 0) - (a.confidence || 0)).slice(0, 1);

  const out = [];
  pool
    .slice()
    .sort((a, b) => (b.confidence || 0) - (a.confidence || 0))
    .forEach(t => {
      const duplicate = out.some(o => {
        if (o.label !== t.label) return false;
        const dAng = Math.abs((t.angle - o.angle + 180) % 360 - 180);
        const dFt = Math.abs((t.distance - o.distance) * M2FT);
        // Wider distance window to dedup same object — but angular separation
        // (>20°) means different object, so always show even if same label.
        return dAng < 20 && dFt < 3.5;
      });
      if (!duplicate) out.push(t);
    });
  return out.slice(0, 6);
}

function radarRangeFt(threats) {
  const dists = [];
  threats.forEach(t => {
    // Receding-only threats don't drive zoom — they're leaving
    if (t.receding && !t.approaching) return;
    const dist = Number(t.distance);
    if (Number.isFinite(dist)) dists.push(dist * M2FT);
    (t.predicted || []).forEach(p => {
      if (p.impact) return;
      const predicted = Number(p.dist_m);
      if (Number.isFinite(predicted)) dists.push(predicted * M2FT);
    });
  });
  if (!dists.length) return DEFAULT_RADAR_RANGE_FT;
  dists.sort((a, b) => a - b);
  // Use median-of-nearest-two to dampen single-frame distance jitter
  const anchor = dists.length >= 2 ? (dists[0] + dists[1]) / 2 : dists[0];
  const farthest = dists[dists.length - 1];
  // 3.5× multiplier: less aggressive zoom, less sensitive to small distance changes
  const focus = Math.max(MIN_RADAR_RANGE_FT, anchor * 3.5);
  const ceiling = Math.min(MAX_RADAR_RANGE_FT, farthest * 1.3);
  return Math.min(ceiling, focus);
}

function RadarPanel({ threats, beams, heading, paused, sweepEnabled, onThreatClick }) {
  const canvasRef = useRef(null);
  const wrapRef   = useRef(null);
  const panelRef  = useRef(null);
  const [size, setSize] = useState(480);
  const [displayRange, setDisplayRange] = useState(DEFAULT_RADAR_RANGE_FT);
  const sweepRef  = useRef(0);
  const animRef   = useRef(0);
  const visibleThreats = useMemo(() => radarThreats(threats), [threats]);

  const threatsRef     = useRef(visibleThreats);
  const beamsRef       = useRef(beams);
  const headingRef     = useRef(heading);
  const sweepEnRef     = useRef(sweepEnabled);
  const pausedRef      = useRef(paused);
  useEffect(()=>{ threatsRef.current  = visibleThreats; }, [visibleThreats]);
  useEffect(()=>{ beamsRef.current    = beams;         }, [beams]);
  useEffect(()=>{ headingRef.current  = heading;       }, [heading]);
  useEffect(()=>{ sweepEnRef.current  = sweepEnabled;  }, [sweepEnabled]);
  useEffect(()=>{ pausedRef.current   = paused;        }, [paused]);

  useEffect(()=>{
    const el = wrapRef.current; if(!el) return;
    const updateSize = (width) => {
      const panel = panelRef.current?.getBoundingClientRect();
      const heightLimit = panel ? panel.height - 104 : window.innerHeight * 0.52;
      const available = Math.min(width, Math.max(280, heightLimit));
      const next = Math.round(Math.min(500, Math.max(260, available)));
      setSize(current => current === next ? current : next);
    };
    const ro = new ResizeObserver(([e])=>updateSize(e.contentRect.width));
    ro.observe(el);
    const onResize = () => updateSize(el.getBoundingClientRect().width);
    window.addEventListener("resize", onResize, { passive: true });
    return ()=>{
      window.removeEventListener("resize", onResize);
      ro.disconnect();
    };
  },[]);

  const rangeRef       = useRef(displayRange);
  const targetRangeRef = useRef(displayRange);
  const threatPoolRef  = useRef({});

  // Target range is updated inside draw() at 60fps from pool distances.

  // Sync display label + hit overlay at ~8Hz — avoids per-frame re-renders
  useEffect(()=>{
    const id = setInterval(()=>setDisplayRange(rangeRef.current), 120);
    return ()=>clearInterval(id);
  }, []);

  const draw = useCallback((dt=16)=>{
    const c = canvasRef.current; if(!c) return;
    const threats    = threatsRef.current;
    const beams      = beamsRef.current;
    const heading    = headingRef.current;
    const sweepOn    = sweepEnRef.current;
    const dpr        = window.devicePixelRatio || 1;
    const css        = size;
    if(c.width !== css*dpr){ c.width = css*dpr; c.height = css*dpr; c.style.width=css+"px"; c.style.height=css+"px"; }
    const ctx = c.getContext("2d");
    ctx.setTransform(dpr,0,0,dpr,0,0);
    ctx.clearRect(0,0,css,css);

    const cx = css/2, cy = css/2;
    const maxR = css * 0.445;
    const toFt = m => m * M2FT;
    const now  = Date.now();
    const RFT = rangeRef.current;

    // ── Radar disc ──
    ctx.beginPath(); ctx.arc(cx,cy,maxR+1,0,Math.PI*2);
    ctx.fillStyle = "#202733"; ctx.fill();

    // Vignette
    const vig = ctx.createRadialGradient(cx,cy,0,cx,cy,maxR);
    vig.addColorStop(0,"rgba(15,157,138,0.12)");
    vig.addColorStop(0.6,"rgba(15,157,138,0.04)");
    vig.addColorStop(1,"rgba(0,0,0,0)");
    ctx.fillStyle=vig; ctx.beginPath(); ctx.arc(cx,cy,maxR,0,Math.PI*2); ctx.fill();

    // ── Rings ──
    const rStep = RFT<=8?1:RFT<=20?5:RFT<=40?10:15;
    const rings = [];
    for(let ft=rStep; ft<=RFT; ft+=rStep) rings.push(Math.round(ft));

    rings.forEach((ft,i)=>{
      const r=(ft/RFT)*maxR, last=i===rings.length-1;
      ctx.beginPath(); ctx.arc(cx,cy,r,0,Math.PI*2);
      ctx.strokeStyle = last?"rgba(15,157,138,0.30)":"rgba(15,23,42,0.08)";
      ctx.lineWidth   = last?1:0.5;
      ctx.stroke();
    });

    // ── Diagonal spokes ──
    ctx.strokeStyle="rgba(15,157,138,0.05)"; ctx.lineWidth=0.5;
    [45,135,225,315].forEach(a=>{
      const rad=(a-90)*Math.PI/180;
      ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(cx+Math.cos(rad)*maxR,cy+Math.sin(rad)*maxR); ctx.stroke();
    });

    // ── Cardinal axes (dashed teal) ──
    ctx.setLineDash([2,6]); ctx.strokeStyle="rgba(15,157,138,0.16)"; ctx.lineWidth=0.7;
    ctx.beginPath(); ctx.moveTo(cx,cy-maxR); ctx.lineTo(cx,cy+maxR); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx-maxR,cy); ctx.lineTo(cx+maxR,cy); ctx.stroke();
    ctx.setLineDash([]);

    // ── Ring distance labels (right side only, pill bg) ──
    ctx.font="bold 8px 'JetBrains Mono',monospace"; ctx.textAlign="left"; ctx.textBaseline="middle";
    rings.forEach((ft,i)=>{
      if(rings.length>4 && i%2!==0) return;
      const r=(ft/RFT)*maxR, lbl=ft+"ft";
      const lx=cx+6, ly=cy-r;
      const tw=ctx.measureText(lbl).width;
      ctx.fillStyle="rgba(255,255,255,0.84)"; ctx.fillRect(lx-2,ly-6,tw+4,12);
      ctx.fillStyle="rgba(15,23,42,0.82)"; ctx.fillText(lbl,lx,ly);
    });

    // ── Audio beam wedges ──
    const beamDirs=[["front",beams.front,0],["right",beams.right,90],["back",beams.back,180],["left",beams.left,270]];
    beamDirs.forEach(([,e,deg])=>{
      const half=22*Math.PI/180, ang=(deg-90)*Math.PI/180, r=maxR*0.94;
      const op=0.03+e*0.12;
      const g=ctx.createRadialGradient(cx,cy,0,cx,cy,r);
      g.addColorStop(0,`rgba(15,157,138,${(op*1.5).toFixed(3)})`);
      g.addColorStop(1,"rgba(15,157,138,0)");
      ctx.fillStyle=g; ctx.beginPath(); ctx.moveTo(cx,cy); ctx.arc(cx,cy,r,ang-half,ang+half); ctx.closePath(); ctx.fill();
      if(e>0.2){
        ctx.strokeStyle=`rgba(40,181,212,${(e*0.35).toFixed(3)})`; ctx.lineWidth=0.8;
        ctx.beginPath(); ctx.arc(cx,cy,r*0.95,ang-half,ang+half); ctx.stroke();
      }
    });

    // ── Compass labels ──
    ctx.font="600 9px 'Inter',sans-serif"; ctx.textAlign="center"; ctx.textBaseline="middle";
    ctx.fillStyle="rgba(15,157,138,0.72)";
    ctx.fillText("FWD",  cx,       cy-maxR-13);
    ctx.fillText("REAR", cx,       cy+maxR+13);
    ctx.fillText("L",    cx-maxR-13, cy);
    ctx.fillText("R",    cx+maxR+13, cy);

    // ── Sweep ──
    if(sweepOn){
      const startA=(sweepRef.current-90)*Math.PI/180;
      const steps=30;
      for(let i=0;i<steps;i++){
        const tv=i/steps;
        const a0=startA-(50*Math.PI/180)*tv;
        const a1=startA-(50*Math.PI/180)*(tv+1/steps);
        ctx.fillStyle=`rgba(15,157,138,${((1-tv)*0.13).toFixed(3)})`;
        ctx.beginPath(); ctx.moveTo(cx,cy); ctx.arc(cx,cy,maxR,a0,a1,true); ctx.closePath(); ctx.fill();
      }
      ctx.strokeStyle="rgba(15,157,138,0.55)"; ctx.lineWidth=1.2;
      ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(cx+Math.cos(startA)*maxR,cy+Math.sin(startA)*maxR); ctx.stroke();
    }

    // ── Threat pool: stable identity + 60fps position lerp ──
    // Matches incoming SSE threats to existing pool entries by label+proximity,
    // lerps rendered position toward target so blips glide rather than snap.
    {
      const TAU_MS  = 200; // position time constant — reaches ~63% in 200ms
      const FADE_MS = 1400;
      const lerpK   = 1 - Math.exp(-dt / TAU_MS);
      const pool    = threatPoolRef.current;

      for (const p of Object.values(pool)) p._seen = false;

      for (const t of threats) {
        let bestKey = null, bestScore = Infinity;
        for (const [pk, p] of Object.entries(pool)) {
          if (p._seen || p.label !== t.label) continue;
          const dA = Math.abs(((t.angle - p.angle) + 540) % 360 - 180);
          const dD = Math.abs(t.distance - p.dist);
          if (dA < 28 && dD < 5) {
            const score = dA + dD * 2;
            if (score < bestScore) { bestScore = score; bestKey = pk; }
          }
        }
        if (bestKey) {
          const p = pool[bestKey];
          p._seen = true; p.lastSeen = now;
          p.targetDist = t.distance; p.targetAngle = t.angle;
          p.urgency = t.urgency; p.confidence = t.confidence;
          p.approaching = t.approaching; p.receding = t.receding;
          p.coasting = t.coasting; p.crossing = t.crossing;
          p.predicted = t.predicted || []; p.vDist = t.vDist || 0; p.vAngle = t.vAngle || 0;
          p.velocity = t.velocity; p.eta = t.eta; p.type = t.type; p.source = t.source;
        } else {
          const key = t.label + '_' + Math.random().toString(36).slice(2);
          pool[key] = {
            key, label: t.label, type: t.type,
            dist: t.distance, angle: t.angle,
            targetDist: t.distance, targetAngle: t.angle,
            vDist: t.vDist||0, vAngle: t.vAngle||0,
            bornAt: now, lastSeen: now,
            urgency: t.urgency, confidence: t.confidence,
            approaching: t.approaching, receding: t.receding,
            coasting: t.coasting, crossing: t.crossing,
            predicted: t.predicted||[], velocity: t.velocity, eta: t.eta,
            source: t.source, _seen: true,
          };
        }
      }

      for (const [pk, p] of Object.entries(pool)) {
        if (!p._seen && now - p.lastSeen > FADE_MS) { delete pool[pk]; continue; }
        const dA = ((p.targetAngle - p.angle) + 540) % 360 - 180;
        p.angle = (p.angle + dA * lerpK + 360) % 360;
        p.dist  = Math.max(0.1, p.dist + (p.targetDist - p.dist) * lerpK);
      }

      threatPoolRef._FADE_MS = FADE_MS;

      targetRangeRef.current = 10;
    }

    // ── Threats ──
    Object.values(threatPoolRef.current).forEach(t=>{
      ctx.setLineDash([]); ctx.lineWidth=1; ctx.globalAlpha=1;

      const FADE_MS = threatPoolRef._FADE_MS || 1400;
      const age     = now - t.bornAt;
      const fadeAge = now - t.lastSeen;
      let alpha = Math.min(1, age / 250);
      if (!t._seen) alpha *= Math.max(0, 1 - fadeAge / FADE_MS);
      if (t.coasting) alpha *= 0.4;

      const r=(Math.min(RFT,toFt(t.dist))/RFT)*maxR;
      const angRad=(t.angle-90)*Math.PI/180;
      const tx=cx+Math.cos(angRad)*r, ty=cy+Math.sin(angRad)*r;

      // urgency → color + dot size
      const urgCfg = {
        critical: { fill:"#dd4d5b", stroke:"rgba(221,77,91,0.5)",  r:12 },
        high:     { fill:"#f08a24", stroke:"rgba(240,138,36,0.35)", r:10 },
        medium:   { fill:"#28b5d4", stroke:"rgba(40,181,212,0.25)",  r:8  },
        low:      { fill:"#728195", stroke:"rgba(114,129,149,0.15)", r:7 },
      }[t.urgency] || { fill:"#728195", stroke:"rgba(114,129,149,0.15)", r:7 };

      // ── Prediction trail ──
      if(t.predicted && t.predicted.length>0){
        const pts=t.predicted.filter(p=>!p.impact).map(p=>{
          const pr=(Math.min(RFT,toFt(p.dist_m))/RFT)*maxR;
          const pa=(p.angle_deg-90)*Math.PI/180;
          return {x:cx+Math.cos(pa)*pr, y:cy+Math.sin(pa)*pr, conf:p.conf||0.5};
        });
        if(pts.length>0){
          const tRGB = t.approaching?"220,70,70": t.receding?"10,200,140":"140,155,150";
          const tOp  = t.approaching?0.70: t.receding?0.45:0.25;

          // Soft glow behind trail for approaching
          if(t.approaching && pts.length>0){
            ctx.strokeStyle=`rgba(${tRGB},${(tOp*0.25*alpha).toFixed(2)})`;
            ctx.lineWidth=7; ctx.setLineDash([]);
            ctx.beginPath(); ctx.moveTo(tx,ty);
            pts.forEach((p,i)=>{ if(i<pts.length-1){const mx=(p.x+pts[i+1].x)/2,my=(p.y+pts[i+1].y)/2; ctx.quadraticCurveTo(p.x,p.y,mx,my);} else ctx.lineTo(p.x,p.y);});
            ctx.stroke();
          }

          // Centerline
          ctx.strokeStyle=`rgba(${tRGB},${(tOp*alpha).toFixed(2)})`;
          ctx.lineWidth=t.approaching?2:1.2;
          ctx.setLineDash(t.receding?[4,5]:[]);
          ctx.beginPath(); ctx.moveTo(tx,ty);
          if(pts.length===1){ ctx.lineTo(pts[0].x,pts[0].y); }
          else { for(let i=0;i<pts.length-1;i++){const mx=(pts[i].x+pts[i+1].x)/2,my=(pts[i].y+pts[i+1].y)/2; ctx.quadraticCurveTo(pts[i].x,pts[i].y,mx,my);} ctx.lineTo(pts[pts.length-1].x,pts[pts.length-1].y); }
          ctx.stroke(); ctx.setLineDash([]);

          // Ghost dots
          pts.forEach((p,i)=>{
            const conf=p.conf??(1-i/pts.length*0.7);
            const dR=Math.max(1.5,4*conf), dA=conf*0.55*alpha;
            ctx.beginPath(); ctx.arc(p.x,p.y,dR,0,Math.PI*2);
            ctx.fillStyle=`rgba(${tRGB},${dA.toFixed(2)})`; ctx.fill();
          });
        }

        // Impact X marker
        const impact=t.predicted.find(p=>p.impact);
        if(impact){
          const iRad=(impact.angle_deg-90)*Math.PI/180;
          // impact.dist_m=0 → collision at user position (center); small offset for visibility
          const iR=Math.max(8,(Math.min(RFT,toFt(impact.dist_m||0))/RFT)*maxR);
          const ix=cx+Math.cos(iRad)*iR, iy=cy+Math.sin(iRad)*iR, xs=6;
          ctx.strokeStyle=`rgba(255,60,60,${(0.9*alpha).toFixed(2)})`;
          ctx.lineWidth=2.5; ctx.setLineDash([]);
          ctx.beginPath(); ctx.moveTo(ix-xs,iy-xs); ctx.lineTo(ix+xs,iy+xs);
          ctx.moveTo(ix+xs,iy-xs); ctx.lineTo(ix-xs,iy+xs); ctx.stroke();
          if(impact.dt_s<5){
            ctx.font="700 8px system-ui,sans-serif"; ctx.textAlign="center"; ctx.textBaseline="bottom";
            ctx.fillStyle=`rgba(255,100,100,${(0.95*alpha).toFixed(2)})`;
            ctx.fillText(impact.dt_s.toFixed(1)+"s", ix, iy-8);
          }
        }
      }

      // ── Approach pulse ──
      if(t.approaching){
        const pulse=1+0.4*Math.sin(now/140);
        ctx.beginPath(); ctx.arc(tx,ty,(urgCfg.r+6)*pulse,0,Math.PI*2);
        ctx.fillStyle=`rgba(220,70,70,${(0.16*alpha).toFixed(3)})`; ctx.fill();
      }
      if(t.urgency==="critical" && !t.approaching){
        const pulse=1+0.25*Math.sin(now/220);
        ctx.beginPath(); ctx.arc(tx,ty,(urgCfg.r+4)*pulse,0,Math.PI*2);
        ctx.fillStyle=`rgba(220,70,70,${(0.10*alpha).toFixed(3)})`; ctx.fill();
      }

      // Outer urgency glow ring
      ctx.beginPath(); ctx.arc(tx,ty,urgCfg.r+3,0,Math.PI*2);
      ctx.strokeStyle=urgCfg.stroke.replace("0.5","0.45").replace("0.35","0.3").replace("0.25","0.2").replace("0.15","0.1");
      ctx.lineWidth=4; ctx.stroke();

      // ── Threat shape ──
      ctx.beginPath();
      if(t.crossing){
        const s=urgCfg.r;
        ctx.moveTo(tx,ty-s); ctx.lineTo(tx+s,ty); ctx.lineTo(tx,ty+s); ctx.lineTo(tx-s,ty); ctx.closePath();
      } else if(t.receding){
        const s=urgCfg.r;
        const tip={x:tx+Math.cos(angRad)*s*1.5, y:ty+Math.sin(angRad)*s*1.5};
        const base={x:tx+Math.cos(angRad)*(-s*0.3), y:ty+Math.sin(angRad)*(-s*0.3)};
        ctx.moveTo(tip.x,tip.y);
        ctx.lineTo(base.x+Math.cos(angRad+Math.PI/2)*s, base.y+Math.sin(angRad+Math.PI/2)*s);
        ctx.lineTo(base.x+Math.cos(angRad-Math.PI/2)*s, base.y+Math.sin(angRad-Math.PI/2)*s);
        ctx.closePath();
      } else {
        ctx.arc(tx,ty,urgCfg.r,0,Math.PI*2);
      }

      if(t.receding){
        ctx.fillStyle=`rgba(15,157,138,${(0.12*alpha).toFixed(3)})`;
        ctx.strokeStyle=`rgba(15,157,138,${(0.90*alpha).toFixed(3)})`;
        ctx.lineWidth=1.8; ctx.fill(); ctx.stroke();
      } else {
        // Fill with urgency color + inner highlight
        ctx.fillStyle=urgCfg.fill+Math.round(alpha*230).toString(16).padStart(2,"0");
        ctx.fill();
        // Inner highlight top
        const hi=ctx.createRadialGradient(tx-urgCfg.r*0.3,ty-urgCfg.r*0.3,0,tx,ty,urgCfg.r);
        hi.addColorStop(0,`rgba(255,255,255,${(0.25*alpha).toFixed(2)})`);
        hi.addColorStop(1,"rgba(255,255,255,0)");
        ctx.fillStyle=hi; ctx.fill();
        // Border
      ctx.strokeStyle=`rgba(31,41,55,${(0.28*alpha).toFixed(3)})`;
        ctx.lineWidth=1; ctx.stroke();
      }

      // ── Velocity arrow ──
      ctx.setLineDash([]); ctx.lineWidth=1;
      if(t.approaching && Math.abs(t.vDist||0)>0.08){
        const al=Math.min(24,Math.abs(t.vDist)*8);
        const ex=tx+Math.cos(angRad+Math.PI)*al, ey=ty+Math.sin(angRad+Math.PI)*al;
        ctx.strokeStyle=`rgba(255,80,80,${(0.95*alpha).toFixed(2)})`; ctx.lineWidth=2.5;
        ctx.beginPath(); ctx.moveTo(tx,ty); ctx.lineTo(ex,ey); ctx.stroke();
        ctx.fillStyle=`rgba(255,80,80,${(0.95*alpha).toFixed(2)})`;
        ctx.beginPath();
        ctx.moveTo(ex,ey);
        ctx.lineTo(ex-Math.cos(angRad+Math.PI-0.42)*7,ey-Math.sin(angRad+Math.PI-0.42)*7);
        ctx.lineTo(ex-Math.cos(angRad+Math.PI+0.42)*7,ey-Math.sin(angRad+Math.PI+0.42)*7);
        ctx.closePath(); ctx.fill();
      } else if(t.receding && Math.abs(t.vDist||0)>0.08){
        const al=Math.min(22,Math.abs(t.vDist)*7);
        const ex=tx+Math.cos(angRad)*al, ey=ty+Math.sin(angRad)*al;
        ctx.strokeStyle=`rgba(15,157,138,${(0.7*alpha).toFixed(2)})`; ctx.lineWidth=1.5;
        ctx.setLineDash([3,3]); ctx.beginPath(); ctx.moveTo(tx,ty); ctx.lineTo(ex,ey); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle=`rgba(15,157,138,${(0.7*alpha).toFixed(2)})`;
        ctx.beginPath();
        ctx.moveTo(ex,ey);
        ctx.lineTo(ex-Math.cos(angRad-0.42)*6,ey-Math.sin(angRad-0.42)*6);
        ctx.lineTo(ex-Math.cos(angRad+0.42)*6,ey-Math.sin(angRad+0.42)*6);
        ctx.closePath(); ctx.fill();
      } else if(t.crossing && t.vAngle){
        const al=Math.min(18,Math.abs(t.vAngle)*0.28), dir=t.vAngle>0?1:-1;
        const prp=angRad+Math.PI/2;
        const ex=tx+Math.cos(prp)*al*dir, ey=ty+Math.sin(prp)*al*dir;
        ctx.strokeStyle=urgCfg.fill+Math.round(alpha*180).toString(16).padStart(2,"0");
        ctx.lineWidth=2; ctx.beginPath(); ctx.moveTo(tx,ty); ctx.lineTo(ex,ey); ctx.stroke();
        ctx.fillStyle=urgCfg.fill+Math.round(alpha*180).toString(16).padStart(2,"0");
        ctx.beginPath();
        ctx.moveTo(ex,ey);
        ctx.lineTo(ex-Math.cos(prp-0.45)*5*dir,ey-Math.sin(prp-0.45)*5*dir);
        ctx.moveTo(ex,ey);
        ctx.lineTo(ex-Math.cos(prp+0.45)*5*dir,ey-Math.sin(prp+0.45)*5*dir);
        ctx.stroke();
      }

      // ── Label ──  distance pill below dot, type above
      const ftStr = toFt(t.dist).toFixed(0)+"ft";
      const typeStr = t.type==="person"?"PERSON": t.type==="vehicle"?"VEHICLE":"OBJ";

      // Distance pill
    ctx.font="700 8px 'JetBrains Mono',monospace"; ctx.textAlign="center"; ctx.textBaseline="middle";
      const dw=ctx.measureText(ftStr).width;
      const pillY=ty+urgCfg.r+10;
      ctx.fillStyle="rgba(255,255,255,0.84)";
      ctx.beginPath();
      if(ctx.roundRect) ctx.roundRect(tx-dw/2-4,pillY-6,dw+8,13,3);
      else ctx.rect(tx-dw/2-4,pillY-6,dw+8,13);
      ctx.fill();
      ctx.fillStyle=urgCfg.fill+Math.round(alpha*230).toString(16).padStart(2,"0");
      ctx.fillText(ftStr, tx, pillY);

      // Type label above (only for larger urgencies)
      if(t.urgency==="critical"||t.urgency==="high"){
        ctx.font="600 7px 'JetBrains Mono',monospace";
        ctx.fillStyle=`rgba(15,23,42,${(0.70*alpha).toFixed(2)})`;
        ctx.fillText(typeStr, tx, ty-urgCfg.r-8);
      }
    });

    ctx.setLineDash([]); ctx.lineWidth=1; ctx.globalAlpha=1;

    // ── User dot (center) ──
    // Heading tick — tiny line pointing forward
    if(Math.abs(heading)>1){
      const hRad=(heading-90)*Math.PI/180;
      const tickLen=14;
      const g=ctx.createLinearGradient(cx,cy, cx+Math.cos(hRad)*tickLen, cy+Math.sin(hRad)*tickLen);
      g.addColorStop(0,"rgba(15,157,138,0.65)");
      g.addColorStop(1,"rgba(15,157,138,0)");
      ctx.strokeStyle=g; ctx.lineWidth=2; ctx.lineCap="round";
      ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(cx+Math.cos(hRad)*tickLen,cy+Math.sin(hRad)*tickLen); ctx.stroke();
      ctx.lineCap="butt";
    }

    // Outer ring
    ctx.beginPath(); ctx.arc(cx,cy,11,0,Math.PI*2);
    ctx.strokeStyle="rgba(15,157,138,0.28)"; ctx.lineWidth=1; ctx.stroke();

    // Inner dot
    ctx.beginPath(); ctx.arc(cx,cy,5,0,Math.PI*2);
    const udg=ctx.createRadialGradient(cx-1,cy-1,0,cx,cy,5);
    udg.addColorStop(0,"#65ead6"); udg.addColorStop(1,"#0f9d8a");
    ctx.fillStyle=udg; ctx.fill();
    ctx.strokeStyle="rgba(40,181,212,0.22)"; ctx.lineWidth=1; ctx.stroke();

    // YOU label
    ctx.font="600 8px 'JetBrains Mono',monospace"; ctx.textAlign="center"; ctx.textBaseline="middle";
    ctx.fillStyle="rgba(15,157,138,0.68)";
    ctx.fillText("YOU", cx, cy-20);

  },[size]);

  useEffect(()=>{
    let last=performance.now();
    function tick(ts){
      const dt=ts-last; last=ts;
      if(!pausedRef.current && sweepEnRef.current)
        sweepRef.current=(sweepRef.current+dt*(360/5000))%360;
      // Smooth range lerp — zoom in ~2s, zoom out ~6s
      const target = targetRangeRef.current;
      const cur    = rangeRef.current;
      const diff   = target - cur;
      if(Math.abs(diff) > 0.05){
        const base = 0.995; // τ≈200ms — matches threat pool lerp so dots stay at fixed radar fraction
        const k    = 1 - Math.pow(base, dt);
        rangeRef.current = Math.min(MAX_RADAR_RANGE_FT, Math.max(MIN_RADAR_RANGE_FT, cur + diff * k));
      }
      draw(dt);
      animRef.current=requestAnimationFrame(tick);
    }
    animRef.current=requestAnimationFrame(tick);
    return ()=>cancelAnimationFrame(animRef.current);
  },[draw]);

  const hits = useMemo(()=>{
    const hcx=size/2, hcy=size/2, hmR=size*0.445, RFT=displayRange;
    return visibleThreats.map(t=>{
      const r=(Math.min(RFT,t.distance*M2FT)/RFT)*hmR;
      const a=(t.angle-90)*Math.PI/180;
      return {id:t.id, threat:t, x:hcx+Math.cos(a)*r, y:hcy+Math.sin(a)*r};
    });
  },[size, visibleThreats, displayRange]);

  return (
    <div className="cell area-radar" ref={panelRef}>
      <div className="panel-hd">
        <div className="ttl"><span className="tag">▸</span>Spatial Radar</div>
        <div className="meta">
          <span>{Math.round(displayRange)}ft</span>
          <span>·</span>
          <span className="live">LIVE</span>
        </div>
      </div>
      <div className="radar-wrap" ref={wrapRef}>
        <canvas ref={canvasRef} className="radar-canvas" width={size} height={size}/>
        {hits.map(h=>(
          <div key={h.id} className="threat-hit"
            style={{left:`calc(50% - ${size/2-h.x}px)`, top:`calc(${h.y+14}px)`}}
            title={`${h.threat.label} · ${(h.threat.distance*3.28084).toFixed(0)}ft`}
            onClick={()=>onThreatClick(h.threat)}
          />
        ))}
      </div>
      <div className="radar-foot">
        <div className="rf">
          <div className="k">Escape</div>
          <div className="v teal">{Math.abs(heading)>2?Math.round(((heading%360)+360)%360).toString().padStart(3,"0")+"°":"—"}</div>
        </div>
        <div className="rf">
          <div className="k">Tracking</div>
          <div className="v">{visibleThreats.length}<span className="u" style={{fontSize:10,color:"var(--tx-3)",marginLeft:4}}>obj</span></div>
        </div>
        <div className="rf">
          <div className="k">Critical</div>
          <div className={"v "+(visibleThreats.some(t=>t.urgency==="critical")?"red":"")}>
            {visibleThreats.filter(t=>t.urgency==="critical").length}
          </div>
        </div>
        <div className="rf">
          <div className="k">Closest</div>
          <div className="v">{visibleThreats.length?(Math.min(...visibleThreats.map(t=>t.distance))*3.28084).toFixed(0):"—"}<span className="u" style={{fontSize:10,color:"var(--tx-3)",marginLeft:4}}>ft</span></div>
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { RadarPanel });
