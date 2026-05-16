// radar.jsx — Spatial threat radar (canvas + DOM hit overlay)

const { useRef, useEffect, useMemo, useState, useCallback } = React;

function RadarPanel({ threats, beams, heading, paused, sweepEnabled, onThreatClick, accent }) {
  const canvasRef = useRef(null);
  const wrapRef = useRef(null);
  const [size, setSize] = useState(480);
  const sweepRef = useRef(0);
  const animRef = useRef(0);
  // Refs so draw() never needs to be recreated when live data changes
  const threatsRef = useRef(threats);
  const beamsRef   = useRef(beams);
  const headingRef = useRef(heading);
  const sweepEnabledRef = useRef(sweepEnabled);
  useEffect(()=>{ threatsRef.current = threats; }, [threats]);
  useEffect(()=>{ beamsRef.current   = beams;   }, [beams]);
  useEffect(()=>{ headingRef.current = heading; }, [heading]);
  useEffect(()=>{ sweepEnabledRef.current = sweepEnabled; }, [sweepEnabled]);
  const pausedRef = useRef(paused);
  useEffect(()=>{ pausedRef.current = paused; }, [paused]);

  // Observe wrap size for responsive canvas
  useEffect(()=>{
    const el = wrapRef.current;
    if(!el) return;
    const ro = new ResizeObserver(([entry])=>{
      const w = entry.contentRect.width;
      const s = Math.min(520, Math.max(280, w));
      setSize(s);
    });
    ro.observe(el);
    return ()=>ro.disconnect();
  },[]);

  const draw = useCallback(()=>{
    const c = canvasRef.current; if(!c) return;
    // Read live data from refs — stable closure, no rAF restarts on data changes
    const threats = threatsRef.current;
    const beams   = beamsRef.current;
    const heading = headingRef.current;
    const sweepEnabled = sweepEnabledRef.current;
    const dpr = window.devicePixelRatio || 1;
    const css = size;
    if(c.width !== css*dpr){ c.width = css*dpr; c.height = css*dpr; c.style.width = css+"px"; c.style.height = css+"px"; }
    const ctx = c.getContext("2d");
    ctx.setTransform(dpr,0,0,dpr,0,0);
    ctx.clearRect(0,0,css,css);

    const cx = css/2, cy = css/2;
    const maxR = css*0.46;
    const distMax = 15; // m

    // Background subtle vignette
    const g = ctx.createRadialGradient(cx,cy,0,cx,cy,maxR);
    g.addColorStop(0,"rgba(32,201,151,0.05)");
    g.addColorStop(1,"rgba(0,0,0,0)");
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(cx,cy,maxR,0,Math.PI*2); ctx.fill();

    // Rings
    const rings = [2,5,10,15];
    ctx.lineWidth = 0.5;
    ctx.strokeStyle = "rgba(255,255,255,0.07)";
    rings.forEach(d=>{
      const r = (d/distMax)*maxR;
      ctx.beginPath(); ctx.arc(cx,cy,r,0,Math.PI*2); ctx.stroke();
    });

    // Quadrant dividers (45/135/225/315)
    ctx.strokeStyle = "rgba(255,255,255,0.045)";
    ctx.lineWidth = 0.5;
    for(let a=45;a<360;a+=90){
      const rad = (a-90)*Math.PI/180;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + Math.cos(rad)*maxR, cy + Math.sin(rad)*maxR);
      ctx.stroke();
    }

    // Cross axes
    ctx.strokeStyle = "rgba(32,201,151,0.18)";
    ctx.setLineDash([2,4]);
    ctx.beginPath(); ctx.moveTo(cx, cy-maxR); ctx.lineTo(cx, cy+maxR); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx-maxR, cy); ctx.lineTo(cx+maxR, cy); ctx.stroke();
    ctx.setLineDash([]);

    // Distance ring labels
    ctx.fillStyle = "rgba(170,170,170,0.55)";
    ctx.font = "9px 'IBM Plex Mono',monospace";
    ctx.textAlign = "left"; ctx.textBaseline = "middle";
    rings.forEach(d=>{
      const r = (d/distMax)*maxR;
      ctx.fillText(d+"m", cx+4, cy - r);
    });

    // (compass labels drawn AFTER wedges — see below)

    // Audio beam wedges (35,145,215,325) — drawn UNDER labels later
    const beamEntries = [
      ["front", beams.front, 35],
      ["right", beams.right, 145],
      ["back",  beams.back,  215],
      ["left",  beams.left,  325],
    ];
    beamEntries.forEach(([k,e,centerDeg])=>{
      const half = 22*Math.PI/180; // 44 degree cone
      const ang = (centerDeg-90)*Math.PI/180;
      const r = maxR*0.95;
      const op = 0.05 + e*0.18;
      // Gradient fade outward
      const grad = ctx.createRadialGradient(cx,cy,0,cx,cy,r);
      grad.addColorStop(0, `rgba(32,201,151,${(op*1.4).toFixed(3)})`);
      grad.addColorStop(1, `rgba(32,201,151,0)`);
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.arc(cx, cy, r, ang-half, ang+half);
      ctx.closePath();
      ctx.fill();
      // Beam edge stroke
      ctx.strokeStyle = `rgba(32,201,151,${(0.12+e*0.3).toFixed(3)})`;
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx+Math.cos(ang-half)*r, cy+Math.sin(ang-half)*r);
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx+Math.cos(ang+half)*r, cy+Math.sin(ang+half)*r);
      ctx.stroke();
      // Energy arc at outer edge
      if(e>0.15){
        ctx.strokeStyle = `rgba(0,229,255,${(e*0.5).toFixed(3)})`;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.arc(cx, cy, r*0.96, ang-half, ang+half);
        ctx.stroke();
      }
    });

    // Compass labels (on top of beams)
    ctx.fillStyle = "rgba(207,207,207,0.95)";
    ctx.font = "600 10px 'IBM Plex Mono',monospace";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText("FRONT", cx, cy - maxR - 12);
    ctx.fillText("BACK",  cx, cy + maxR + 12);
    ctx.fillText("LEFT",  cx - maxR - 18, cy);
    ctx.fillText("RIGHT", cx + maxR + 20, cy);

    // Sweep (rotating gradient)
    if(sweepEnabled){
      const sweep = sweepRef.current;
      const startA = (sweep - 90)*Math.PI/180;
      const endA = (sweep - 90 - 40)*Math.PI/180;
      const grd = ctx.createConicGradient ? ctx.createConicGradient(startA, cx, cy) : null;
      // Use polygonal trail fallback
      const steps = 24;
      for(let i=0;i<steps;i++){
        const t = i/steps;
        const a0 = startA - (40*Math.PI/180)*t;
        const a1 = startA - (40*Math.PI/180)*(t+1/steps);
        const alpha = (1-t)*0.18;
        ctx.fillStyle = `rgba(32,201,151,${alpha.toFixed(3)})`;
        ctx.beginPath();
        ctx.moveTo(cx,cy);
        ctx.arc(cx,cy,maxR,a0,a1,true);
        ctx.closePath();
        ctx.fill();
      }
      // Leading edge
      ctx.strokeStyle = "rgba(0,229,255,0.55)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cx,cy);
      ctx.lineTo(cx+Math.cos(startA)*maxR, cy+Math.sin(startA)*maxR);
      ctx.stroke();
    }

    // Escape direction arrow (from Mac planner / stereo)
    if(heading !== 0){
      const hRad = (heading - 90)*Math.PI/180;
      const arrowLen = maxR * 0.72;
      const ex = cx+Math.cos(hRad)*arrowLen, ey = cy+Math.sin(hRad)*arrowLen;
      // Glow trail
      ctx.strokeStyle = "rgba(32,201,151,0.15)";
      ctx.lineWidth = 10;
      ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(ex,ey); ctx.stroke();
      // Main line
      ctx.strokeStyle = "#20C997";
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(ex,ey); ctx.stroke();
      // Arrowhead
      const headLen = 10, headAngle = 0.42;
      ctx.fillStyle = "#20C997";
      ctx.beginPath();
      ctx.moveTo(ex, ey);
      ctx.lineTo(ex - headLen*Math.cos(hRad-headAngle), ey - headLen*Math.sin(hRad-headAngle));
      ctx.lineTo(ex - headLen*Math.cos(hRad+headAngle), ey - headLen*Math.sin(hRad+headAngle));
      ctx.closePath(); ctx.fill();
      // Label
      ctx.font = "bold 9px 'IBM Plex Mono',monospace";
      ctx.fillStyle = "rgba(32,201,151,0.9)";
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      const lx = cx+Math.cos(hRad)*(arrowLen+14), ly = cy+Math.sin(hRad)*(arrowLen+14);
      ctx.fillText("GO", lx, ly);
    }

    // Threats (fade in/out by age)
    const now = Date.now();
    threats.forEach(t=>{
      const age = now - t.bornAt;
      let alpha = 1;
      if(age < 200) alpha = age/200;
      else if(age > t.ttl - 300) alpha = Math.max(0,(t.ttl - age)/300);
      if(t.coasting) alpha *= 0.4;
      // Extrapolate position using velocity × time since last server update
      const dtS = Math.min((now - (t._rx || now)) / 1000, 0.15); // cap at 150ms
      const smoothDist  = Math.max(0.3, t.distance + (t.vDist  || 0) * dtS);
      const smoothAngle = ((t.angle   + (t.vAngle || 0) * dtS) + 360) % 360;
      const r = (Math.min(distMax, smoothDist)/distMax)*maxR;
      const angRad = (smoothAngle - 90)*Math.PI/180;
      const tx = cx + Math.cos(angRad)*r;
      const ty = cy + Math.sin(angRad)*r;
      const col = {critical:"#E24B4A", high:"#BA7517", medium:"#FFD54F", low:"#888888"}[t.urgency];

      // Predicted trajectory trail — draw before the main dot so it's underneath
      if(t.predicted && t.predicted.length > 0){
        // Dashed line along predicted path
        ctx.setLineDash([2,3]);
        ctx.strokeStyle = `rgba(255,255,255,0.18)`;
        ctx.lineWidth = 0.8;
        ctx.beginPath();
        ctx.moveTo(tx, ty);
        t.predicted.forEach(p=>{
          const pr = (Math.min(distMax, p.dist_m)/distMax)*maxR;
          const pa = (p.angle_deg - 90)*Math.PI/180;
          ctx.lineTo(cx + Math.cos(pa)*pr, cy + Math.sin(pa)*pr);
        });
        ctx.stroke();
        ctx.setLineDash([]);
        // Ghost dots at each predicted position
        t.predicted.forEach((p, i)=>{
          const pr = (Math.min(distMax, p.dist_m)/distMax)*maxR;
          const pa = (p.angle_deg - 90)*Math.PI/180;
          const px = cx + Math.cos(pa)*pr, py = cy + Math.sin(pa)*pr;
          const ghostAlpha = 0.35 - i*0.07;
          ctx.beginPath();
          ctx.arc(px, py, 4, 0, Math.PI*2);
          ctx.fillStyle = col + Math.round(Math.max(0,ghostAlpha)*255).toString(16).padStart(2,"0");
          ctx.fill();
        });
      }

      // Outer pulse for critical
      if(t.urgency==="critical"){
        const pulse = 1 + 0.4*Math.sin(now/180);
        ctx.beginPath();
        ctx.arc(tx, ty, 14*pulse, 0, Math.PI*2);
        ctx.fillStyle = `rgba(226,75,74,${(0.18*alpha).toFixed(3)})`;
        ctx.fill();
      }
      // Shape: diamond for crossing, circle for approaching/static
      ctx.beginPath();
      if(t.crossing){
        const s = 11;
        ctx.moveTo(tx, ty-s); ctx.lineTo(tx+s, ty);
        ctx.lineTo(tx, ty+s); ctx.lineTo(tx-s, ty); ctx.closePath();
      } else {
        ctx.arc(tx, ty, 11, 0, Math.PI*2);
      }
      ctx.fillStyle = col + Math.round(alpha*255).toString(16).padStart(2,"0");
      ctx.fill();
      ctx.strokeStyle = `rgba(0,0,0,${(0.5*alpha).toFixed(3)})`;
      ctx.lineWidth = 1;
      ctx.stroke();

      // Crossing: draw lateral arrow to show direction of travel
      if(t.crossing && t.vAngle){
        const arrowLen = Math.min(20, Math.abs(t.vAngle) * 0.3);
        const dir = t.vAngle > 0 ? 1 : -1;  // positive vAngle = moving clockwise
        const perpRad = angRad + Math.PI/2;  // perpendicular to radial = tangential
        ctx.beginPath();
        ctx.moveTo(tx, ty);
        const ex = tx + Math.cos(perpRad)*arrowLen*dir;
        const ey = ty + Math.sin(perpRad)*arrowLen*dir;
        ctx.lineTo(ex, ey);
        ctx.strokeStyle = col + Math.round(alpha*200).toString(16).padStart(2,"0");
        ctx.lineWidth = 2;
        ctx.stroke();
        // arrowhead
        const ah = 5;
        ctx.beginPath();
        ctx.moveTo(ex, ey);
        ctx.lineTo(ex - Math.cos(perpRad-0.5)*ah*dir, ey - Math.sin(perpRad-0.5)*ah*dir);
        ctx.moveTo(ex, ey);
        ctx.lineTo(ex - Math.cos(perpRad+0.5)*ah*dir, ey - Math.sin(perpRad+0.5)*ah*dir);
        ctx.stroke();
      }

      // Icon
      ctx.font = "12px serif";
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillStyle = `rgba(0,0,0,${(0.85*alpha).toFixed(3)})`;
      ctx.fillText(t.icon, tx, ty+0.5);

      // Confidence label
      ctx.font = "8.5px 'IBM Plex Mono',monospace";
      ctx.fillStyle = `rgba(255,255,255,${(0.85*alpha).toFixed(3)})`;
      ctx.fillText(t.confidence+"%", tx, ty+18);

      // ID label small (top)
      ctx.fillStyle = `rgba(170,170,170,${(0.55*alpha).toFixed(3)})`;
      ctx.font = "7.5px 'IBM Plex Mono',monospace";
      ctx.fillText(t.id, tx, ty-16);
    });

    // Center user (with ring)
    ctx.beginPath();
    ctx.arc(cx, cy, 9, 0, Math.PI*2);
    ctx.strokeStyle = "rgba(32,201,151,0.35)";
    ctx.lineWidth = 0.5;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, Math.PI*2);
    ctx.fillStyle = "#20C997";
    ctx.fill();
    // User label
    ctx.fillStyle = "rgba(255,255,255,0.55)";
    ctx.font = "8px 'IBM Plex Mono',monospace";
    ctx.fillText("YOU", cx, cy+18);

    // Corner crosshairs
    const tk = 12;
    ctx.strokeStyle = "rgba(32,201,151,0.35)";
    ctx.lineWidth = 0.8;
    [[8,8,1,1],[css-8,8,-1,1],[8,css-8,1,-1],[css-8,css-8,-1,-1]].forEach(([x,y,sx,sy])=>{
      ctx.beginPath();
      ctx.moveTo(x, y); ctx.lineTo(x+tk*sx, y);
      ctx.moveTo(x, y); ctx.lineTo(x, y+tk*sy);
      ctx.stroke();
    });
  },[size]);

  useEffect(()=>{
    let last = performance.now();
    function tick(now){
      const dt = now - last; last = now;
      if(!pausedRef.current && sweepEnabledRef.current){
        sweepRef.current = (sweepRef.current + dt*(360/5000)) % 360;
      }
      draw();
      animRef.current = requestAnimationFrame(tick);
    }
    animRef.current = requestAnimationFrame(tick);
    return ()=>cancelAnimationFrame(animRef.current);
  },[draw]);

  // Hit overlay positions for threats (DOM clickable)
  const hits = useMemo(()=>{
    const cx = size/2, cy = size/2;
    const maxR = size*0.46;
    const distMax = 15;
    return threats.map(t=>{
      const r = (Math.min(distMax, t.distance)/distMax)*maxR;
      const angRad = (t.angle - 90)*Math.PI/180;
      return { id:t.id, threat:t, x: cx + Math.cos(angRad)*r, y: cy + Math.sin(angRad)*r };
    });
  },[size, threats]);

  return (
    <div className="cell area-radar tactical">
      <div className="panel-hd">
        <div className="ttl"><span className="tag">▸</span>SPATIAL RADAR</div>
        <div className="meta">
          <span>15M</span>
          <span>·</span>
          <span className="live">LIVE</span>
        </div>
      </div>
      <div className="radar-wrap" ref={wrapRef}>
        <canvas ref={canvasRef} className="radar-canvas" width={size} height={size} />
        {hits.map(h=>(
          <div key={h.id}
            className="threat-hit"
            style={{left: `calc(50% - ${size/2 - h.x}px)`, top: `calc(${h.y + 14}px)` }}
            title={`${h.threat.label} · ${h.threat.distance.toFixed(1)}m · ${h.threat.confidence}%`}
            onClick={()=>onThreatClick(h.threat)}
          />
        ))}
      </div>
      <div className="radar-foot">
        <div className="rf">
          <div className="k">Escape</div>
          <div className="v teal">{heading !== 0 ? Math.round(((heading%360)+360)%360).toString().padStart(3,"0")+"°" : "—"}</div>
        </div>
        <div className="rf">
          <div className="k">Tracking</div>
          <div className="v">{threats.length}<span className="u" style={{fontSize:10,color:"var(--tx-3)",marginLeft:4}}>obj</span></div>
        </div>
        <div className="rf">
          <div className="k">Critical</div>
          <div className={"v "+(threats.some(t=>t.urgency==="critical")?"red":"")}>
            {threats.filter(t=>t.urgency==="critical").length}
          </div>
        </div>
        <div className="rf">
          <div className="k">Closest</div>
          <div className="v">{threats.length ? Math.min(...threats.map(t=>t.distance)).toFixed(1) : "—"}<span className="u" style={{fontSize:10,color:"var(--tx-3)",marginLeft:4}}>m</span></div>
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { RadarPanel });
