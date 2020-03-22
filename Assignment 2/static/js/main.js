// toggle accordion dropdown
// var acc = document.getElementsByClassName("accordion");
// var i;

// for (i = 0; i < acc.length; i++) {
//   acc[i].addEventListener("click", function() {
//     this.classList.toggle("active");
//     var panel = this.nextElementSibling;
//     if (panel.style.maxHeight) {
//       panel.style.maxHeight = null;
//     } else {
//       panel.style.maxHeight = panel.scrollHeight + "px";
//     } 
//   });
// }

// // change + to - on active
// $(document).ready(function(){
//     // Add minus icon for collapse element which is open by default
//     $(".collapse.show").each(function(){
//         $(this).prev(".card-header").find(".fa").addClass("fa-minus").removeClass("fa-plus");
//     });
    
//     // Toggle plus minus icon on show hide of collapse element
//     $(".collapse").on('show.bs.collapse', function(){
//         $(this).prev(".card-header").find(".fa").removeClass("fa-plus").addClass("fa-minus");
//     }).on('hide.bs.collapse', function(){
//         $(this).prev(".card-header").find(".fa").removeClass("fa-minus").addClass("fa-plus");
//     });
// });