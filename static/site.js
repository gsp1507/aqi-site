// table search + column sort (any table with class "js-table")
document.querySelectorAll('table.js-table').forEach(function (t) {
  var body = t.tBodies[0];
  t.querySelectorAll('th.s').forEach(function (th, idx) {
    var asc = true;
    th.addEventListener('click', function () {
      var col = th.cellIndex;
      var rows = Array.from(body.rows);
      rows.sort(function (a, b) {
        var x = a.cells[col].dataset.v || a.cells[col].textContent.trim();
        var y = b.cells[col].dataset.v || b.cells[col].textContent.trim();
        var nx = parseFloat(x), ny = parseFloat(y);
        var r = (!isNaN(nx) && !isNaN(ny)) ? nx - ny : x.localeCompare(y);
        return asc ? r : -r;
      });
      asc = !asc;
      rows.forEach(function (r) { body.appendChild(r); });
    });
  });
});
function filterTable(tableId, q, selects) {
  var t = document.getElementById(tableId);
  var text = (q.value || '').toLowerCase();
  Array.from(t.tBodies[0].rows).forEach(function (r) {
    var ok = r.textContent.toLowerCase().indexOf(text) > -1;
    selects.forEach(function (s) {
      if (s.value && r.dataset[s.dataset.key] !== s.value) ok = false;
    });
    r.style.display = ok ? '' : 'none';
  });
  var c = document.getElementById(tableId + '-count');
  if (c) c.textContent = Array.from(t.tBodies[0].rows).filter(function (r) { return r.style.display !== 'none'; }).length;
}
